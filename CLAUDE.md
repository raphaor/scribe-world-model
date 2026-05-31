# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Handwriting World Model (HWM) for historical document OCR. A research codebase that iterates on a JEPA-inspired (Joint Embedding Predictive Architecture) recognizer for handwritten text lines from ALTO-segmented page scans, with CTC as the supervised head. Many model versions (v2 → v17) coexist; each is a different experiment, not a deprecation chain.

See `README.md` for the high-level architecture diagram and loss equations.

## Common commands

```bash
# Train (default = v5, mixed mode, 30 epochs). Resolves --alto-dirs from
# config.ALTO_DIRS. Saves to hwm_v<N>.pt (path defined per-version in
# model_registry.REGISTRY[ver].save_path).
python train.py --model-version v17

# Train a specific version with overrides
python train.py --model-version v17 --epochs 50 --batch-size 16 --lambda-sigreg 0

# Pure self-supervised adaptation (no labels)
python train.py --model-version v11 --mode adapt --save-path hwm_v11_adapt.pt

# Resume / fine-tune from a checkpoint (auto phase-switches if --mode differs)
python train.py --model-version v11 --checkpoint hwm_v11_adapt.pt --mode full

# CER eval on the seeded val split (use the same --model-version as training)
python recognize.py --model hwm_v17.pt --model-version v17

# Single-image / synthetic-data inference
python inference.py --model hwm_model.pt --num-tests 5

# Ketos baseline (Lectaurep clone) — runs the official kraken pipeline
python scripts/ketos_train_baseline.py --step compile --alto-dirs <...> --output-dir <...>
```

No test suite, no linter, no formatter configured. `requirements.txt` is incomplete — `kraken` is required (used by `data_alto.py` for ALTO XML parsing and line polygon extraction) but is not pinned.

## Architecture overview

### Per-version registry (`model_registry.py`)

`model_registry.REGISTRY` is the single source of truth for everything that varies by `--model-version`: image height, collate style, window/stride, bucketing, AMP/LR overrides, checkpoint path, and a `builder(args, num_classes)` callable.

`train.py` should never branch on the version string. To add a new version: write the model class in `model.py`, add a `_build_vN` in `model_registry.py`, and append a `ModelSpec` to `REGISTRY`. The argparse choices, dataset config, and CTC build all read from the registry automatically.

`default_train_args()` synthesizes the argparse defaults so inference scripts (`recognize.py`, `inference.py`) can call the same builders without a CLI parse.

### Config per version (`config.py`)

Every model has its own block of `*_V<N>` constants. **Do not generalize** older constants into newer ones — keeping each version's hyperparameters frozen is intentional so old checkpoints stay reproducible. When tweaking a value mid-experiment, edit only the active version's block.

Notable per-version constants:
- `IMG_HEIGHT_V<N>` — line image height (32 px for v1–v3, 48 for v3–v4, 120 for v5+).
- `LAMBDA_CTC_V<N>`, `LAMBDA_JEPA_V<N>`, `LAMBDA_SIGREG_V<N>` — loss weights (CLI flags `--lambda-pred`, `--lambda-sigreg` override per-call).
- `LECTAUREP_*` — v15 clone of the official lectaurep_base ketos recipe (pure CTC, no Transformer/JEPA/SIGReg).

### Training loop (`train.py`)

Three modes:
- `full`: supervised (prediction + SIGReg + CTC).
- `adapt`: self-supervised only (no labels).
- `mixed` (default): interleaves one full and one adapt batch per step.

Key training-time behaviors that are easy to miss:
- **Phase switch**: when `--mode` differs from the checkpoint's saved `mode` (or `--phase-restart`), the epoch counter, optimizer, and scheduler are reset and a 2-epoch linear warmup is auto-enabled. This avoids inheriting the previous phase's cosine decay.
- **Discriminative LR**: encoder + SSL trunk uses `lr * encoder_lr_mult` (default 0.1); CTC head uses full `lr`. Some specs force `encoder_lr_mult=1.0` (e.g. v15/v16/v17) via `force_encoder_lr_mult`.
- **AMP**: enabled on CUDA by default; some specs force `--no-amp` via `force_no_amp` (v15/v16/v17, models with large intermediate dims).
- **Bucketing**: `LengthBucketBatchSampler` groups variable-width lines (v5+ `collate_style="v5"`) to bound peak VRAM. Disabled with `--no-bucket`.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is set before `import torch`. Do not move this — it must be set before the CUDA allocator initializes.
- **No `cudnn.benchmark`**: input widths vary every batch; benchmarking would re-cache workspaces per shape and fragment VRAM.
- **No `itertools.cycle`** on adapt_loader: it caches every batch in memory across the epoch.

### Loss aggregation (`loss_bundle.py`)

Newer models (v12+) use `LossBundle` — a list of `LossTerm(name, weight, fn)` summed into a single scalar + metrics dict. A term whose `fn(ctx)` returns `None` is silently skipped, which is how the same bundle drives both supervised and `adapt` steps without branching. To add a loss: write the compute fn, append a `LossTerm` to the bundle (see `make_v12_bundle`).

### Data (`data_alto.py`)

- `AltoLineDataset` parses ALTO XML + matching `.jpg` via Kraken; extracts line polygons, resizes to `img_height`, stores as uint8 to keep RAM low.
- Parsed datasets are pickled into `.cache_alto/dataset_<sha>.pkl` keyed by directory set + height + width cap. **Do not delete cache files mid-run** without checking — see [[feedback_destructive_ops_during_runs]] in user memory; a training may be reading them.
- Train/val split is seeded (`generator=Generator().manual_seed(42)`) — reproducible across runs. Validation augmentation is disabled via a shallow `copy.copy` of the dataset (shares the `samples` list, separate `augment` flag).

### Evaluation (`recognize.py`)

`evaluate_cer` takes a seeded random subset (`_CER_SEED = 770414`) when `max_samples` is set — **not** the first-N batches. Loader-order CER is biased: the width-bucket sampler iterates shortest-first, so first-N would sample only easy lines. The fixed seed keeps train/val CER stable across epochs.

The 10-line example display (`_show_fixed_samples`) uses a different seed (`_SAMPLE_SEED = 20240517`) and temporarily disables augmentation on the underlying dataset.

## Conventions

- Checkpoint files live at repo root as `hwm_v<N>*.pt`. Use `--save-path` to keep adapt/full/baseline runs from overwriting each other (e.g. `hwm_v11_adapt.pt`, `hwm_v11_full.pt`).
- French-language comments and CLI help are fine and expected (mixed FR/EN throughout).
- Checkpoint loading is shape-tolerant: layers with mismatched shapes are skipped with a warning, and optimizer state is restored only when the full state-dict matches.
