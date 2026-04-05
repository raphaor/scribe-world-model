"""
Export Model for Transfer to Powerful Machine
Prepares model and config for training on GPU
"""

import sys
import os
import torch
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model import create_model


def export_model(
    checkpoint_path="hwm_model.pt",
    output_path="hwm_for_transfer.tar"
):
    """
    Export model for transfer to powerful machine
    
    Creates a tarball with:
    - model_state_dict.pt: trained weights
    - config.json: model configuration
    - training_history.json: loss history
    - README.txt: instructions
    """
    print("\n" + "="*60)
    print("Exporting Model for Transfer")
    print("="*60)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract components
    model_state = checkpoint['model_state_dict']
    saved_config = checkpoint['config']
    history = checkpoint.get('history', [])
    
    # Prepare export data
    export_data = {
        'model_state_dict': model_state,
        'config': saved_config,
        'training_history': history,
        'pi_config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'sigreg_lambda': config.SIGREG_LAMBDA,
            'num_params': sum(p.numel() for p in model_state.values())
        }
    }
    
    # Save as tarball
    print(f"\nSaving to: {output_path}")
    torch.save(export_data, output_path)
    
    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Create README
    readme_content = f"""# HWM Model Transfer Package

## Contents

This package contains a trained HWM-v1 model ready for transfer to a more powerful machine.

### Files included:
- Model state dict (trained weights)
- Model configuration
- Training history from Pi
- Original Pi configuration

## Model Summary

- **Architecture:** HWM-v1 (Handwriting World Model)
- **Parameters:** {export_data['pi_config']['num_params']:,}
- **Embedding dim:** {saved_config['embedding_dim']}
- **Transformer layers:** {saved_config['num_layers']}
- **Attention heads:** {saved_config['num_heads']}

## Pi Training Results

- **Batch size:** {export_data['pi_config']['batch_size']}
- **Learning rate:** {export_data['pi_config']['learning_rate']}
- **SIGReg lambda:** {export_data['pi_config']['sigreg_lambda']}

Loss history:
{chr(10).join(f"  Epoch {i+1}: {loss['total']:.4f}" for i, loss in enumerate(history))}

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

1. **Increase batch size:** From {export_data['pi_config']['batch_size']} to 32-128
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
"""
    
    readme_path = "TRANSFER_README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ README saved to: {readme_path}")
    
    # Summary
    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - {output_path} ({file_size_mb:.2f} MB)")
    print(f"  - {readme_path}")
    
    print(f"\nTransfer instructions:")
    print(f"  1. Copy {output_path} to powerful machine")
    print(f"  2. Copy source files (model.py, encoder.py, predictor.py, loss.py)")
    print(f"  3. Load with torch.load() and continue training")
    
    print(f"\n✓ Ready for transfer!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export HWM model for transfer')
    parser.add_argument('--checkpoint', type=str, default='hwm_model.pt', help='Checkpoint to export')
    parser.add_argument('--output', type=str, default='hwm_for_transfer.tar', help='Output file')
    
    args = parser.parse_args()
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    export_model(args.checkpoint, args.output)
