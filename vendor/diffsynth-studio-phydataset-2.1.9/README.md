# DiffSynth-Studio-phydataset 2.1.9 snapshot

This directory preserves the exact DiffSynth-Studio source snapshot used by the PhysEdit Wan5B training jobs.

## Provenance

- Production source path: `/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora/repos/DiffSynth-Studio-phydataset`
- Declared package version: `2.1.9`
- Upstream project: [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- License: Apache-2.0; the copied upstream license is included as `LICENSE`
- Git commit: unavailable because the production snapshot did not contain `.git` metadata
- Official PyPI wheel used for the comparison: `diffsynth-2.1.9-py3-none-any.whl`
- Official wheel SHA256: `be98adf71f1d42ede52fc699330c23bc7ceaa5d72283aeb87cfc4a330869e5d0`

No weights, videos, checkpoints, logs, or training outputs are included.

## Files

- `source.tar.gz`: buildable source subset containing `diffsynth/`, Wan training examples, tests, tools, sanitized dependency lock data, documentation, and the Apache license. The generated lock report containing production cache paths is intentionally excluded.
- `critical/operators.py`: exact patched data operator used in production.
- `critical/wan_train.py`: exact Wan training entry point showing `sampling_strategy="uniform_full_timeline"`.
- `critical/test_phydataset_wan.py`: PhysEdit training-state tests from the snapshot.
- `patches/uniform_full_timeline.patch`: focused patch against the official DiffSynth 2.1.9 wheel.

## Integrity

```text
aff3c5275cf71356d3f9ac4a79ee264893b64427cb39196381d6bd3714ae40eb  source.tar.gz
7b29bff027cc232c3b509e0892da34bc25f5b6759a1b52383b44a56776cb91b8  critical/operators.py
ad57214c95b6275e14a0d8e0cf43a09687aa177d8852ab46b2d6ec68d0d5d069  critical/wan_train.py
8356fbdbe8749c24febb284df7d2d03daf31d1d5905f77995b9a1e5f17c151a5  critical/test_phydataset_wan.py
```

## `uniform_full_timeline`

Official DiffSynth 2.1.9 decodes the first `num_frames` frames. The production patch adds a strategy switch:

- `prefix`: preserves official behavior.
- `uniform_full_timeline`: selects `num_frames` indices uniformly over the complete source timeline, including both endpoints.

For `T` raw frames and `N` requested frames, the selected indices are:

```python
[round(i * (T - 1) / (N - 1)) for i in range(N)]
```

The integer formula is deterministic across NumPy versions. The patch also supports optional JSONL sampling traces through `PHYDATASET_VIDEO_TRACE`.

The production Wan loader activates the strategy in `UnifiedDataset.default_video_operator(...)`:

```python
sampling_strategy="uniform_full_timeline"
```

## Apply only the focused patch

From an unpacked official DiffSynth 2.1.9 source tree:

```bash
git apply /path/to/patches/uniform_full_timeline.patch
```

Then ensure your Wan dataset operator passes `sampling_strategy="uniform_full_timeline"`, as shown in `critical/wan_train.py`.

## Use the complete snapshot

```bash
tar -xzf source.tar.gz
cd DiffSynth-Studio-phydataset
pip install -e .
```

The archive root contains the package files directly; choose any extraction directory name you prefer.
