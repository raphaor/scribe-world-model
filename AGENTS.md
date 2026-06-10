# AGENTS.md

Quick-reference for agents working in this repo. For full architecture details, see `CLAUDE.md`.

## Commands

```bash
python train.py --model-version v17                          # Train (mixed mode, 30 epochs)
python train.py --model-version v17 --epochs 50 --batch-size 16 --lambda-sigreg 0
python train.py --model-version v11 --mode adapt --save-path hwm_v11_adapt.pt   # Self-supervised only
python train.py --model-version v11 --checkpoint hwm_v11_adapt.pt --mode full   # Resume / phase-switch
python recognize.py --model hwm_v17.pt --model-version v17   # CER eval
python inference.py --model hwm_model.pt --num-tests 5       # Single-image inference
```

No test suite, linter, or formatter. No CI.

## Critical gotchas

- `train.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **before** `import torch` (line 35). Do not move it or remove it — it must land before the CUDA allocator initializes.
- `cudnn.benchmark` is intentionally disabled: variable-width inputs would re-cache per shape and fragment VRAM.
- Do not use `itertools.cycle` on `adapt_loader` — it caches every batch in memory.
- `.cache_alto/*.pkl` files are live-read during training. Do not delete them mid-run.
- `*.pt` checkpoint files are gitignored. They live at repo root as `hwm_v<N>*.pt`.

## Architecture in 30 seconds

- `model_registry.py` (`REGISTRY` dict) is the single source of truth per `--model-version`. Never branch on version strings in `train.py`.
- To add a version: model class in `model.py`, `_build_vN` + `ModelSpec` in `model_registry.py`. Argparse choices and dataset config update automatically.
- `config.py` has per-version constant blocks (`*_V<N>`). **Do not generalize** old blocks into new ones — each version's hyperparameters are frozen for checkpoint reproducibility.
- Training modes: `full` (supervised), `adapt` (self-supervised), `mixed` (default, interleaves both).
- Losses (v12+): `loss_bundle.py` — `LossBundle` of `LossTerm(name, weight, fn)`. Returning `None` from a fn silently skips it (drives both supervised and adapt without branching).

## Key dependencies

`requirements.txt` is incomplete. `kraken` is required by `data_alto.py` for ALTO XML parsing but is not pinned.

## Conventions

- French-language comments and CLI help are expected (mixed FR/EN).
- Checkpoint loading is shape-tolerant: mismatched layers are skipped with a warning; optimizer state restores only on full state-dict match.
- Train/val split is seeded (`manual_seed(42)`). CER eval uses a different fixed seed (`_CER_SEED = 770414`) with random subset sampling, not first-N.
- Use `--save-path` to keep different training phases from overwriting each other (e.g. `hwm_v11_adapt.pt`, `hwm_v11_full.pt`).
