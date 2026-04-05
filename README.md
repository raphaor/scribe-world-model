# HWM-v1: Handwriting World Model - Proof of Concept

**Lightweight implementation for Raspberry Pi (ARM64)**

## Overview

HWM-v1 is a "World Model 1D" adapted for handwriting recognition, inspired by LeWorldModel (LeWM). This PoC validates the architecture on a Raspberry Pi with limited resources.

### Architecture

```
Input: Image line (H×W)
  ↓
[CNN 1D Encoder] → Embedding z_t ∈ R^64
  ↓
[Transformer Predictor] (2 layers, 2 heads)
  ↓
Prediction ż_{t+1}
  ↓
Loss: MSE(z_{t+1}, ż_{t+1}) + λ × SIGReg(Z)
```

### Key Features

- ✅ **CPU-only** training (no GPU required)
- ✅ **Lightweight**: ~100K-500K parameters (under 1M limit)
- ✅ **Pi-friendly**: Batch size 2-4, ~6.7GB RAM available
- ✅ **Self-supervised**: Learns from pixels only (no labels needed)
- ✅ **SIGReg**: Prevents collapse without EMA/stop-gradient

## Quick Start

### 1. Setup

```bash
cd ~/projects/hwm-poc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Test Components

```bash
# Test encoder
python encoder.py

# Test predictor
python predictor.py

# Test loss functions
python loss.py

# Test complete model
python model.py

# Test data generation
python generate_data.py
```

### 3. Train (Lightweight)

```bash
# Quick training (2 epochs, 100 synthetic lines)
python train_light.py --epochs 2 --num-lines 100

# Adjust parameters if needed
python train_light.py --epochs 3 --batch-size 2 --num-lines 50
```

### 4. Test Inference

```bash
# Test on new synthetic data
python inference.py --model hwm_model.pt --num-tests 5
```

### 5. Export for Transfer

```bash
# Export model for powerful machine
python export_model.py --checkpoint hwm_model.pt --output hwm_for_transfer.tar
```

## Project Structure

```
hwm-poc/
├── config.py           # Hyperparameters
├── encoder.py          # CNN 1D encoder
├── predictor.py        # Transformer predictor
├── loss.py             # Prediction loss + SIGReg
├── model.py            # Complete HWM-v1 model
├── generate_data.py    # Synthetic data generator
├── train_light.py      # Lightweight training script
├── inference.py        # Inference and testing
├── export_model.py     # Export for transfer
├── requirements.txt    # Dependencies
├── README.md           # This file
├── data/
│   └── synthetic/      # Generated images
└── hwm_model.pt        # Trained model
```

## Configuration (config.py)

```python
# Architecture
EMBEDDING_DIM = 64          # vs 192 in LeWM
NUM_LAYERS = 2              # Minimal transformer
NUM_HEADS = 2               # Small attention
BATCH_SIZE = 4              # Pi memory limit
SEQ_LEN = 50                # Short sequences

# Loss
SIGREG_LAMBDA = 0.1         # Regularization strength

# Data
NUM_SYNTHETIC_LINES = 100   # Small for PoC
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

## Validation Criteria

**Success = ✅**
- [x] Model instantiates without error
- [x] Forward pass works
- [x] Loss decreases on 1-2 epochs
- [x] Predictions are valid (no NaN)
- [x] Model exportable

**What we're NOT doing:**
- ❌ Training a production model
- ❌ Using large datasets (IAM = 1GB+)
- ❌ Achieving high accuracy
- ❌ Running for many epochs

**Goal:** Validate architecture, ensure loss decreases, verify numerical stability.

## Training on Pi

### Expected Performance

- **1 epoch** (~100 lines, batch_size=4): ~30-60 seconds
- **2 epochs**: ~1-2 minutes
- **Memory usage**: ~200-500MB (well under 6.7GB limit)

### Monitoring

Training logs show:
- Total loss
- Prediction loss (MSE)
- SIGReg regularization loss
- Progress per batch

Look for **decreasing total loss** across epochs.

## Transfer to Powerful Machine

### What to Transfer

1. **Model weights**: `hwm_for_transfer.tar`
2. **Source files**: `model.py`, `encoder.py`, `predictor.py`, `loss.py`
3. **README**: `TRANSFER_README.md` (auto-generated)

### Recommended Scaling

On a machine with GPU:

```python
# Increase capacity
EMBEDDING_DIM = 192      # 3x larger
NUM_LAYERS = 4-6         # More depth
NUM_HEADS = 4-8          # More attention
BATCH_SIZE = 32-128      # Much larger batches

# Use real data
dataset = IAMHandwritingDatabase()  # ~5800 lines

# Train longer
num_epochs = 50-100
```

### Next Steps After Transfer

1. **Scale up model** (more layers, higher dims)
2. **Use real data** (IAM, RIMES, etc.)
3. **Train for many epochs** (50-100)
4. **Evaluate on validation set**
5. **Compare vs baselines** (Kraken, CRNN)
6. **Test applications**:
   - Recognition (linear probe)
   - Anomaly detection
   - Style transfer

## Technical Details

### SIGReg Loss

**Simple Isometric Gaussian Regularization**

From LeWorldModel - forces embeddings to follow isotropic Gaussian distribution:
- Penalizes off-diagonal covariance (decorrelation)
- Penalizes deviation from unit variance
- Prevents collapse without EMA or stop-gradient

```python
L_SIGReg = ||Cov(Z) - I||² + ||Var(Z) - 1||²
```

### Encoder Architecture

Very lightweight:
- Input: (H×W) flattened columns
- 2 linear layers (256 → 128 → 64)
- ReLU + Dropout
- Output: z ∈ R^64

### Predictor Architecture

Minimal transformer:
- Positional encoding
- 2-layer TransformerEncoder
- 2 attention heads
- Output projection to R^64

## Troubleshooting

### Memory Error

```bash
# Reduce batch size
python train_light.py --batch-size 2

# Reduce data
python train_light.py --num-lines 50
```

### Loss Explodes

```python
# In config.py, reduce learning rate
LEARNING_RATE = 5e-4  # from 1e-3

# Or reduce SIGReg weight
SIGREG_LAMBDA = 0.05  # from 0.1
```

### NaN Values

Check:
1. Input data is normalized [0, 1]
2. Learning rate not too high
3. Gradient clipping enabled (default: max_norm=1.0)

## References

- **LeWorldModel**: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" (arxiv:2603.19312)
- **SIGReg**: From LeWM paper
- **Kraken**: OCR system for handwriting recognition

## Author

PoC developed on Raspberry Pi 4 (ARM64) - April 2026

---

**Status**: ✅ Architecture validated, ready for scaling on powerful hardware
