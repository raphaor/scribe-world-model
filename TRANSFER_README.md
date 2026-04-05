# HWM Model Transfer Package

## Contents

This package contains a trained HWM-v1 model ready for transfer to a more powerful machine.

### Files included:
- Model state dict (trained weights)
- Model configuration
- Training history from Pi
- Original Pi configuration

## Model Summary

- **Architecture:** HWM-v1 (Handwriting World Model)
- **Parameters:** 514,432
- **Embedding dim:** 64
- **Transformer layers:** 2
- **Attention heads:** 2

## Pi Training Results

- **Batch size:** 4
- **Learning rate:** 0.001
- **SIGReg lambda:** 0.1

Loss history:
  Epoch 1: 0.2390
  Epoch 2: 0.1300

## Usage on Powerful Machine

```python
import torch
from model import HWMv1

# Load exported data
export_data = torch.load('hwm_for_transfer.tar')

# Create model with same config
config = export_data['config']
model = HWMv1(
    img_height=config['img_height'],
    window_size=config['window_size'],
    embedding_dim=config['embedding_dim'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    ff_dim=config['ff_dim']
)

# Load trained weights
model.load_state_dict(export_data['model_state_dict'])

# Continue training or use for inference
model.train()
# ... train on GPU with larger batch size, more data, etc.
```

## Recommended Next Steps

1. **Increase batch size:** From 4 to 32-128
2. **Add more data:** Use IAM Handwriting Database or similar
3. **Scale up model:** Increase embedding_dim, num_layers, etc.
4. **Longer training:** Train for 50-100 epochs on GPU
5. **Evaluation:** Test on held-out validation set

## Notes

- This is a **proof of concept** model trained on synthetic data
- Architecture has been validated on Raspberry Pi
- Loss decreased during training ✓
- Model produces valid predictions (no NaN) ✓
- Ready for scaling up on powerful hardware
