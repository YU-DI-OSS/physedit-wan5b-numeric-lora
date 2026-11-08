# Wan5B transferable adapters and checkpoint index

This index records the production paths and integrity data needed to continue PhysEdit Wan2.2-TI2V-5B training on another server.

Snapshot time: `2026-11-08T08:01:10Z`.

## Step5275 inference/initialization adapter

GitHub LFS path:

```text
artifacts/weights/wan5b-step5275-adapter.safetensors
```

Production source:

```text
/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora/runs/wan5b/mixed13109_resume5250_v1/output/step-5275.safetensors
```

Properties:

- Size: `161287952` bytes.
- SHA256: `8b1df92f54e4dadbff0e91cf2bc7503fa233abfdfc3e7820f62e2d87b1786cf5`.
- Tensor count: `600` LoRA tensors.
- LoRA rank: `32`.
- Intended use: LoRA initialization or inference. This file does not contain optimizer, scheduler, RNG, or sampler state.

After cloning the repository:

```bash
git lfs pull --include="artifacts/weights/wan5b-step5275-adapter.safetensors"
sha256sum artifacts/weights/wan5b-step5275-adapter.safetensors
```

The corresponding exact-resume full state is on shared storage:

```text
/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora/runs/wan5b/mixed13109_resume5250_v1/fullstate/step-00005275/
```

Its sampler position is `11766 / 13112`, with `global_step = 5275`. See `step5275/metadata.json` and `step5275/SHA256SUMS`.

## M1 latest verified full state

Newest file-level verified checkpoint:

```text
/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora/runs/wan5b-h3final-v2-20261107T0630Z/m1/new-media-8000-v1/fullstate/step-00000001/
```

- Size: approximately `462 MiB`.
- Progress: `global_step = 1`, sampler position `4 / 8000`.
- Contains LoRA, optimizer, scheduler, RNG, and data cursor in `adapter.pt` and `training_state.pt`.
- `COMPLETE` exists and every file matches `m1-latest/verification.json`.
- The later training process exited after progressing beyond this checkpoint. No traceback, OOM, disk-quota, or no-space signature was found in the inspected log tail. Resuming this checkpoint repeats work performed after step 1 because no newer complete state exists.

Conservative completed-stage boundary:

```text
/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora/runs/wan5b-h3final-v2-20261107T0630Z/m1/new-media-4000-v1/fullstate/step-00001000/
```

This boundary has `global_step = 1000`, sampler position `4000 / 4000`, and a clean `COMPLETE` monitor result. Use it when starting the 8000-record stage from cursor zero; use the step-1 checkpoint only for exact continuation of the already-started stage.

## M2 latest verified full state

```text
/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora/runs/wan5b-h3final-v2-20261107T0630Z/m2/a800-gravity-medium-m1paths-v1/fullstate/step-00000001/
```

- Size: approximately `484 MiB`.
- `world_size = 1`, gradient accumulation `4`.
- Contains `600` LoRA tensors plus the Harmonic numeric encoder, optimizer state, scheduler, RNG, rank state, and cursor.
- `COMPLETE` exists and every file matches `m2-latest/verification.json`.
- The later M2 process reached step 1252 before exiting, but no later complete checkpoint was written. Continuing from step 1 therefore repeats post-checkpoint work.
- No traceback, OOM, disk-quota, or no-space signature was found in the inspected log tail.

Do not initialize M2 from the ordinary step5275 adapter if the objective is exact M2 continuation. Use the complete M2 checkpoint so the Harmonic encoder and optimizer state are retained.

## Transfer between servers

On another StarLight server with the same `/XYFS02` mount, do not copy the checkpoint. Point the resume argument directly at the appropriate full-state directory.

For a server without the shared mount, copy the directory as a unit and verify every file before starting:

```bash
rsync -a --partial --info=progress2 SOURCE_HOST:/absolute/fullstate/step-NNNNNNNN/ ./step-NNNNNNNN/
cd step-NNNNNNNN
sha256sum -c /path/to/SHA256SUMS
test "$(cat COMPLETE)" = "ok"
```

Preserve filenames and directory structure. Never copy only `adapter.pt` when an exact continuation is required.

## Required external assets

The checkpoints do not contain the Wan2.2 base model or tokenizer. The original shared paths are:

```text
/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora/models/Wan2.2-TI2V-5B/
/XYFS02/HDD_POOL/sysu_xdliang/sysu_xdliang_3/tianhaixin/phydataset_lora/models/Wan2.2-TI2V-5B/google/umt5-xxl/
```

Base model manifest SHA256 recorded by all listed checkpoints:

```text
96f38705903db47095706f6544cd1eca795a8899ee48b3b508e9e2328f5c71b6
```

H3 manifest SHA256 used by M1/M2:

```text
1ba18226f991fd1f9cd12c19c5a49fc9dfc2ad980712bb929e881d4f9abfeba6
```
