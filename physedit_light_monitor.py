"""Small stdlib-only monitor for one independent Wan5B training stage."""
import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

def utc(ts=None):
    return datetime.fromtimestamp(ts or time.time(), timezone.utc).isoformat()

def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def tail(path, limit=1048576):
    if not path.exists():
        return ''
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size-limit))
        return handle.read().decode('utf-8', 'replace')

def last_loss(path):
    lines = tail(path, 8192).splitlines()
    for line in reversed(lines):
        parts = line.split(',')
        if len(parts) >= 3 and parts[1] == 'loss':
            return int(parts[0]), float(parts[2])
    return 0, None

def latest_checkpoint(root, numeric):
    entries = sorted(root.glob('step-*')) if root.is_dir() else []
    if not entries:
        return None, None
    p = entries[-1]
    required = ('COMPLETE', 'adapter.pt', 'training_state.pt', 'verification.json', 'metadata.json')
    if numeric:
        required += ('rank-00000.pt',)
    problems = []
    for name in required:
        if not (p / name).is_file():
            problems.append('missing ' + name)
    if not problems:
        verification = json.loads((p / 'verification.json').read_text())
        for name in required:
            if name in verification and (p / name).stat().st_size != verification[name]['bytes']:
                problems.append('size mismatch ' + name)
        metadata = json.loads((p / 'metadata.json').read_text())
        if not numeric:
            progress = metadata['progress']
            if p.name != f"step-{progress['global_step']:08d}":
                problems.append('step mismatch')
            if progress['sampler']['position'] > progress['sampler']['dataset_length']:
                problems.append('cursor outside dataset')
        elif not metadata.get('rank_aware') or metadata.get('world_size') != 1:
            problems.append('numeric rank metadata mismatch')
    return str(p), problems

def check(stage, gpu_index, numeric):
    now = time.time()
    monitor = stage / 'monitor'
    monitor.mkdir(exist_ok=True)
    fingerprint = json.loads((stage / 'fingerprint.json').read_text())
    records = int(fingerprint['records'])
    pid = int((stage / 'continuous.pid').read_text().strip())
    alive = True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        alive = False
    except PermissionError:
        alive = True
    step, loss = last_loss(stage / 'output/loss.csv')
    samples = min(records, step * int(fingerprint['world_size']) * int(fingerprint['gradient_accumulation_steps']))
    log = tail(stage / 'stdout-continuous.log')
    regularizer = None
    if numeric:
        values = re.findall(r'"regularizer":\s*([-+0-9.eE]+)', log)
        if values:
            regularizer = float(values[-1])
    gpu = {'memory_mib': None, 'utilization_pct': None}
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], timeout=12, text=True)
        rows = [[int(x.strip()) for x in row.split(',')] for row in out.splitlines()]
        row = next(x for x in rows if x[0] == gpu_index)
        gpu = {'memory_mib': row[1], 'utilization_pct': row[2]}
    except Exception as exc:
        gpu['error'] = str(exc)
    checkpoint, cp_errors = latest_checkpoint(stage / 'fullstate', numeric)
    previous_path = monitor / 'status.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    last_growth = now if step > previous.get('global_step', -1) else previous.get('last_growth_unix', now)
    coverage_path = monitor / 'coverage_config.json'
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    prior = int(coverage.get('prior_covered_samples', 0))
    full_target = coverage.get('full_target_records')
    cumulative = prior + samples
    recent = []
    checks_path = monitor / 'checks.jsonl'
    for line in tail(checks_path, 262144).splitlines():
        try:
            item = json.loads(line)
            if item['checked_unix'] >= now-7200:
                recent.append(item)
        except (ValueError, KeyError):
            pass
    rate = None
    for item in recent:
        elapsed = now-item['checked_unix']
        growth = samples-item.get('consumed_samples', samples)
        if elapsed >= 300 and growth > 0:
            rate = growth/elapsed
            break
    tranche_eta = (records-samples)/rate/3600 if rate and rate>0 else None
    finish = utc(now+tranche_eta*3600) if tranche_eta is not None else None
    rough_full_lower_bound = ((full_target-cumulative)/rate/3600
                              if full_target and rate and rate>0 and cumulative<full_target else None)
    errors = []
    if loss is not None and not math.isfinite(loss):
        errors.append('nonfinite loss')
    if numeric and regularizer is not None and not math.isfinite(regularizer):
        errors.append('nonfinite numeric regularizer')
    if re.search(r'OutOfMemoryError|CUDA out of memory|\bNaN\b|\bInf\b|Traceback \(most recent call last\)', log[-40000:], re.I):
        errors.append('error signature in recent training log')
    if cp_errors:
        p = Path(checkpoint)
        if now - p.stat().st_mtime > 300:
            errors.extend('checkpoint ' + x for x in cp_errors)
    target_steps = math.ceil(records/(int(fingerprint['world_size'])*int(fingerprint['gradient_accumulation_steps'])))
    completed = (step >= target_steps and checkpoint is not None and not cp_errors and
                 Path(checkpoint).name == f'step-{target_steps:08d}')
    if not alive and not completed:
        errors.append('training PID exited before dataset completion')
    if alive and now-last_growth > 2700:
        errors.append('no optimizer-step growth for over 45 minutes')
    if alive and gpu['memory_mib'] is not None and gpu['memory_mib'] < 1000 and now-last_growth > 600:
        errors.append('training GPU memory below 1 GiB')
    status = {'checked_at': utc(now), 'checked_unix': now, 'stage': fingerprint['stage_id'],
              'pid': pid, 'pid_alive': alive, 'status': 'ALERT' if errors else ('COMPLETE' if completed else 'RUNNING'),
              'errors': errors, 'global_step': step, 'consumed_samples': samples,
              'dataset_records': records, 'loss': loss, 'numeric_regularizer': regularizer,
              'recent_samples_per_second': rate, 'tranche_eta_hours': tranche_eta,
              'tranche_expected_finish_utc': finish,
              'cumulative_covered_samples': cumulative,
              'full_target_records': full_target,
              'cumulative_coverage_fraction': cumulative/full_target if full_target else None,
              'rough_full_lower_bound_hours': rough_full_lower_bound,
              'full_eta_quality': 'unavailable: unadmitted media/field bindings and future stage transitions',
              'gpu': gpu, 'latest_complete_checkpoint': checkpoint if not cp_errors else None,
              'checkpoint_errors': cp_errors, 'last_growth_unix': last_growth,
              'loss_log': str(stage / 'output/loss.csv'), 'training_log': str(stage / 'stdout-continuous.log')}
    atomic_json(previous_path, status)
    with (monitor / 'checks.jsonl').open('a') as handle:
        handle.write(json.dumps(status, ensure_ascii=False) + '\n')
    if errors:
        atomic_json(monitor / 'ALERT.json', status)
    else:
        (monitor / 'ALERT.json').unlink(missing_ok=True)
        (monitor / 'MONITOR_FAILURE.json').unlink(missing_ok=True)
    for percent in (25,50,75,100):
        marker = monitor / f'milestone-{percent:03d}.json'
        if samples*100 >= records*percent and not marker.exists():
            atomic_json(marker, status)
    last_report_file = monitor / 'last_report_unix.txt'
    last_report = float(last_report_file.read_text()) if last_report_file.exists() else 0
    last_report_path = monitor / 'latest_report.json'
    prior_report = json.loads(last_report_path.read_text()) if last_report_path.exists() else {}
    prior_rate = prior_report.get('recent_samples_per_second')
    rate_changed = bool(rate and prior_rate and abs(rate-prior_rate)/prior_rate >= 0.30)
    state_changed = bool(prior_report and prior_report.get('status') != status['status'])
    if now-last_report >= 7200 or state_changed or rate_changed:
        report = {'generated_at': utc(now), 'stage': fingerprint['stage_id'], 'status': status['status'],
                  'global_step': step, 'consumed_samples': samples, 'dataset_records': records,
                  'recent_samples_per_second': rate, 'loss': loss, 'numeric_regularizer': regularizer,
                  'latest_complete_checkpoint': status['latest_complete_checkpoint'],
                  'tranche_eta_hours': tranche_eta, 'tranche_expected_finish_utc': finish,
                  'cumulative_covered_samples': cumulative,
                  'full_target_records': full_target,
                  'cumulative_coverage_fraction': status['cumulative_coverage_fraction'],
                  'full_dataset_eta_hours': None,
                  'rough_full_lower_bound_hours': rough_full_lower_bound,
                  'full_dataset_eta_note': status['full_eta_quality']}
        atomic_json(monitor / ('report-' + datetime.fromtimestamp(now, timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'), report)
        atomic_json(monitor / 'latest_report.json', report)
        last_report_file.write_text(str(now) + '\n')
    return status

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', type=Path)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--numeric', action='store_true')
    ap.add_argument('--loop', action='store_true')
    args = ap.parse_args()
    while True:
        try:
            print(json.dumps(check(args.stage, args.gpu, args.numeric), ensure_ascii=False), flush=True)
        except Exception as exc:
            p = args.stage/'monitor'
            p.mkdir(exist_ok=True)
            atomic_json(p/'MONITOR_FAILURE.json', {'at':utc(), 'error':repr(exc)})
            print('MONITOR_FAILURE '+repr(exc), flush=True)
        if not args.loop:
            break
        time.sleep(900)
