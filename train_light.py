"""
Lightweight Training Script for HWM on Raspberry Pi
Trains for a few epochs just to verify loss decreases
"""

import sys
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model import create_model
from generate_data import create_dataloader


def train_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_pred_loss = 0
    total_sigreg_loss = 0
    num_batches = 0
    
    for batch_idx, (img_seqs, texts, lengths) in enumerate(loader):
        # Move to device
        img_seqs = img_seqs.to(device)
        
        # Filter out sequences that are too short
        if img_seqs.shape[1] < 2:
            continue
        
        # Forward pass
        optimizer.zero_grad()
        loss, losses_dict = model.compute_loss(img_seqs)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += losses_dict['total']
        total_pred_loss += losses_dict['pred']
        total_sigreg_loss += losses_dict['sigreg']
        num_batches += 1
        
        # Log progress
        if batch_idx % config.LOG_INTERVAL == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: "
                  f"Loss={losses_dict['total']:.4f} "
                  f"(Pred={losses_dict['pred']:.4f}, "
                  f"SIGReg={losses_dict['sigreg']:.4f})")
    
    # Compute averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_pred = total_pred_loss / num_batches if num_batches > 0 else 0
    avg_sigreg = total_sigreg_loss / num_batches if num_batches > 0 else 0
    
    return {
        'total': avg_loss,
        'pred': avg_pred,
        'sigreg': avg_sigreg
    }


def train(num_epochs=2, batch_size=None, num_lines=None):
    """
    Train HWM model
    Args:
        num_epochs: number of epochs (default 2 for quick test)
        batch_size: override config batch size
        num_lines: number of synthetic lines to generate
    """
    print("\n" + "="*60)
    print("HWM-v1 Training on Raspberry Pi")
    print("="*60)
    
    # Override config if specified
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_lines is None:
        num_lines = config.NUM_SYNTHETIC_LINES
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    print(f"\nCreating model...")
    model = create_model()
    model = model.to(device)
    
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    
    if total_params > config.MAX_PARAMS:
        print(f"⚠️  WARNING: Model exceeds max params limit!")
    
    # Create data loader
    print(f"\nCreating data loader ({num_lines} lines, batch_size={batch_size})...")
    loader = create_dataloader(
        num_lines=num_lines,
        batch_size=batch_size,
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
        shuffle=True
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 60)
    
    start_time = time.time()
    history = []
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{num_epochs}")
        losses = train_epoch(model, loader, optimizer, device, epoch)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Loss: {losses['total']:.4f}")
        print(f"  Avg Pred: {losses['pred']:.4f}")
        print(f"  Avg SIGReg: {losses['sigreg']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        history.append(losses)
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total time: {total_time:.1f}s")
    print(f"\nLoss progression:")
    for i, losses in enumerate(history, 1):
        print(f"  Epoch {i}: {losses['total']:.4f}")
    
    # Check if loss decreased
    if len(history) >= 2:
        if history[-1]['total'] < history[0]['total']:
            print(f"\n✓ Loss DECREASED: {history[0]['total']:.4f} → {history[-1]['total']:.4f}")
        else:
            print(f"\n⚠️  Loss did not decrease significantly")
    
    # Save model
    save_path = "hwm_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'img_height': config.IMG_HEIGHT,
            'window_size': config.WINDOW_SIZE,
            'embedding_dim': config.EMBEDDING_DIM,
            'num_layers': config.NUM_LAYERS,
            'num_heads': config.NUM_HEADS,
            'ff_dim': config.FF_DIM,
        },
        'history': history
    }, save_path)
    print(f"\n✓ Model saved to: {save_path}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Train HWM model')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--num-lines', type=int, default=None, help='Number of synthetic lines')
    
    args = parser.parse_args()
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_lines=args.num_lines
    )
