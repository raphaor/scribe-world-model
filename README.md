# Scribe World Model (HWM)

Handwriting World Model for historical document OCR. A JEPA-inspired architecture (Joint Embedding Predictive Architecture) that learns visual representations of handwriting through next-frame prediction, combined with CTC-based character recognition.

## Architecture

```
Input: Handwriting line image (48 x W pixels, grayscale)
  |
  |  sliding windows (stride=4, width=32)
  v
[Frame_1] [Frame_2] ... [Frame_T]     T frames of 48x32 pixels
  |          |              |
  v          v              v
[  Conv2D Encoder (shared weights)  ]  Each frame -> z in R^128
  |          |              |
  v          v              v
 z_1        z_2    ...     z_T         Latent embedding sequence
  |                         |
  +--- Two heads -----------+
  |                         |
  v                         v
[Transformer Predictor]   [CTC Head]
(causal, predicts z_{t+1}) (character logits)
  |                         |
  v                         v
L_pred + L_SIGReg          L_CTC
```

### Loss components

| Loss | Formula | Role |
|------|---------|------|
| **Prediction** | `MSE(z_pred, z_target)` | Self-supervised: predict next embedding |
| **SIGReg** | `\|Cov(Z) - I\|^2 + \|Var(Z) - 1\|^2` | Prevent embedding collapse (no EMA needed) |
| **CTC** | CTC alignment loss | Supervised: character recognition |

## Training modes

Three training modes control how losses are combined:

### `mixed` (default)

Alternates supervised and self-supervised batches each training step. The encoder learns both to predict future frames (structure of handwriting) and to recognize characters. This is the recommended mode when you have some annotated data and want robust embeddings.

```bash
# Uses annotated ALTO data for both supervised and self-supervised steps
python train.py

# With additional unannotated scans for the self-supervised steps
python train.py --unannotated-dirs D:/scans/lot1 D:/scans/lot2
```

### `full`

Pure supervised training. All three losses active (prediction + SIGReg + CTC). Every batch requires ground truth text.

```bash
python train.py --mode full
```

### `adapt`

Pure self-supervised. Only prediction + SIGReg losses. No ground truth needed. Use this to adapt the encoder to a new handwriting style before fine-tuning with `full` or `mixed`.

```bash
python train.py --mode adapt
```

## Model versions

| | v1 | v2 | v3 (default) |
|---|---|---|---|
| Encoder | MLP | Conv2D (3 layers) | Conv2D (5 layers + BatchNorm) |
| Embedding dim | 64 | 96 | 128 |
| Transformer | 2 layers, 2 heads | 2 layers, 2 heads | 4 layers, 4 heads |
| Window size | 10 | 10 | 32 |
| Stride | 5 | 5 | 4 |
| CTC head | No | Yes | Yes |
| Image height | 32 | 48 | 48 |

```bash
python train.py --model-version v3   # default
python train.py --model-version v2
```

## Data

### ALTO XML format

Training data consists of page scans (JPG) paired with ALTO XML segmentation files. Kraken's parsers extract individual text lines from the pages.

```bash
python train.py --alto-dirs D:/OCR/bars_dordogne D:/OCR/saint_chamassy
```

Datasets are cached in `.cache_alto/` after first load for faster restarts.

### Data augmentation

Applied on-the-fly during training:
- Random rotation (+-3 degrees, 50% chance)
- Random contrast adjustment (0.85-1.15x, 50% chance)
- Gaussian noise (sigma=5, 50% chance)

## Quick start

### Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Train

```bash
# Default: mixed mode, v3, 30 epochs
python train.py --alto-dirs <path_to_alto_data>

# With GPU, larger batch
python train.py --batch-size 32 --lr 1e-3

# Resume from checkpoint
python train.py --checkpoint hwm_v3.pt
```

### Evaluate

```bash
# CER evaluation on ALTO data
python recognize.py --model hwm_v3.pt --alto-dirs <path_to_alto_data>
```

### Inference

```bash
python inference.py --model hwm_model.pt --num-tests 5
```

## CLI reference

```
python train.py [OPTIONS]

--mode {mixed,full,adapt}     Training mode (default: mixed)
--model-version {v2,v3}       Model architecture (default: v3)
--epochs N                    Number of epochs (default: 30)
--batch-size N                Batch size (default: 32)
--lr FLOAT                    Learning rate (default: 1e-3)
--alto-dirs DIR [DIR ...]     Annotated ALTO data directories
--unannotated-dirs DIR [...]  Extra dirs for self-supervised data (mixed mode)
--checkpoint PATH             Resume from checkpoint
--num-workers N               DataLoader workers, 0=main thread (default: 0)
```

## Project structure

```
scribe-world-model/
  config.py          Hyperparameters (v1/v2/v3)
  encoder.py         CNN encoders (MLP, Conv2D, Conv2DV2)
  predictor.py       Causal Transformer predictor
  ctc_head.py        Linear CTC projection head
  loss.py            Prediction + SIGReg + CTC losses
  model.py           HWMv1, HWMv2, HWMv3 models
  data_alto.py       ALTO dataset, unannotated dataset, collate functions
  generate_data.py   Synthetic data generator + sliding window extraction
  train.py           Main training script (mixed/full/adapt)
  train_light.py     Lightweight training (v1, synthetic data)
  recognize.py       CTC greedy decoding + CER evaluation
  inference.py       Inference and testing
  export_model.py    Export model for transfer
  .cache_alto/       Cached parsed datasets
```

## Configuration (config.py)

```python
# v3 (default)
WINDOW_SIZE_V3 = 32
STRIDE_V3 = 4
IMG_HEIGHT_V3 = 48
EMBEDDING_DIM_V3 = 128
NUM_LAYERS_V3 = 4
NUM_HEADS_V3 = 4
FF_DIM_V3 = 384
LAMBDA_CTC_V3 = 2.0
DROPOUT = 0.1
```

## GPU memory management

Training includes several optimizations to prevent GPU memory overflow:
- Explicit tensor cleanup after each training step
- Periodic `torch.cuda.empty_cache()` every 50 batches
- Sequence length capping (`max_seq_len=512`) in collate functions
- `torch.amp` mixed precision (automatic on CUDA)
- `gc.collect()` + cache clear between epochs

If you run out of memory, reduce `--batch-size` or add more `--unannotated-dirs` with shorter lines.

## References

- **LeWorldModel**: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" (arxiv:2603.19312)
- **SIGReg**: Simple Isometric Gaussian Regularization (from LeWM)
- **Kraken**: OCR engine used for ALTO parsing and line extraction
