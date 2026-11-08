# PhysEdit Wan5B Numeric LoRA

Training and data-preparation code used for the PhysEdit Wan2.2-TI2V-5B numeric-enhanced LoRA experiments.

The repository contains the M2 training path built around:

- LoRA rank 32 on attention Q/K/V/O and FFN projections.
- Harmonic embeddings for physical numeric values.
- Numerical Attention regularization.
- Strict gravity-value extraction and UMT5 token binding.
- Full-state checkpointing for LoRA, numeric encoder, optimizer, scheduler, RNG, and data cursor.

## Repository layout

```text
code/
  train_m2_wave01.py              Training entry point
  numeric_runner.py               Numeric-aware training loop
  wan_numeric_conditioning.py     Harmonic numeric encoder
  wan_numeric_production_bridge.py
  wan_numeric_runtime.py          Per-sample numeric metadata runtime
  phydataset_training.py          Full-state checkpoint helpers
  sitecustomize.py                DiffSynth integration hooks

wave01_numeric_schema.py          Physical-value schema and token binding
build_m2_medium_gravity_prompt_v2.py
build_m2_m1paths_v1.py            Strict dataset builders
validate_m2_medium_gravity_prompt_v2.py
validate_m2_m1paths_v1.py         Dataset and token-binding validation
preflight_m2_step114.py           Full checkpoint preflight checks
physedit_light_monitor.py         Lightweight read-only training monitor
```

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- Hugging Face Transformers and Accelerate
- safetensors
- DiffSynth-Studio with Wan2.2 training support
- A local Wan2.2-TI2V-5B checkpoint and UMT5 tokenizer

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

DiffSynth-Studio is an external training dependency and must be installed from the version used by your environment.

## Data format

Each accepted JSONL record contains a video path, Medium or Detailed prompt, and strict numeric supervision metadata. Gravity supervision uses:

- `parameter_type = gravity`
- `type_id = 0`
- SI unit `m/s^2`
- normalization `gravity / 25`
- UMT5 token offsets validated before the 512-token limit

Ambiguous numbers, unrelated counts, truncated values, duplicate videos, and media with mismatched SHA256 are rejected by the builders.

## Environment-specific paths

The checked-in scripts preserve the production defaults used for the original experiment. Before running in another environment, update the dataset, model, tokenizer, DiffSynth, and output paths or wrap them with your own launcher configuration.

## Safety and scope

This repository intentionally excludes model weights, videos, manifests containing private paths, checkpoints, optimizer states, logs, and credentials. It does not include the Wan2.2 base model or DiffSynth-Studio source.

## License

No open-source license has been assigned yet. All rights are reserved unless the repository owner adds a license.
