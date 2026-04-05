"""
Inference Script for HWM
Test prediction on new sequences
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from model import create_model, HWMv1
from generate_data import extract_columns, SyntheticHandwritingDataset


def load_model(checkpoint_path="hwm_model.pt"):
    """Load trained model"""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with saved config
    saved_config = checkpoint.get('config', {})
    model = HWMv1(
        img_height=saved_config.get('img_height', config.IMG_HEIGHT),
        window_size=saved_config.get('window_size', config.WINDOW_SIZE),
        embedding_dim=saved_config.get('embedding_dim', config.EMBEDDING_DIM),
        num_layers=saved_config.get('num_layers', config.NUM_LAYERS),
        num_heads=saved_config.get('num_heads', config.NUM_HEADS),
        ff_dim=saved_config.get('ff_dim', config.FF_DIM),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded ({model.count_parameters():,} parameters)")
    
    return model


def test_prediction(model, img_tensor):
    """
    Test prediction on a single image
    Args:
        model: trained HWM model
        img_tensor: (H, W) image tensor
    Returns:
        embeddings, prediction, error
    """
    # Extract columns
    columns = extract_columns(
        img_tensor,
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE
    )
    
    # Add batch dimension
    columns = columns.unsqueeze(0)  # (1, T, H, W)
    
    with torch.no_grad():
        # Encode
        z_seq = model.encode_sequence(columns)  # (1, T, D)
        
        # Predict next (use all but last)
        if z_seq.shape[1] >= 2:
            z_history = z_seq[:, :-1, :]
            z_pred = model.predictor(z_history)  # (1, D)
            
            # Target is last embedding
            z_target = z_seq[:, -1, :]
            
            # Compute error
            error = torch.nn.functional.mse_loss(z_pred, z_target).item()
        else:
            z_pred = None
            z_target = None
            error = None
    
    return z_seq.squeeze(0), z_pred.squeeze(0) if z_pred is not None else None, error


def run_inference_tests(model, num_tests=5):
    """Run inference tests on synthetic data"""
    print("\n" + "="*60)
    print("Running Inference Tests")
    print("="*60)
    
    # Create test dataset
    dataset = SyntheticHandwritingDataset(num_lines=num_tests, seed=999)
    
    errors = []
    
    for i in range(num_tests):
        img, text = dataset[i]
        
        print(f"\nTest {i+1}/{num_tests}: '{text}'")
        print(f"  Image shape: {img.shape}")
        
        # Test prediction
        embeddings, prediction, error = test_prediction(model, img)
        
        print(f"  Embeddings shape: {embeddings.shape}")
        
        if prediction is not None:
            print(f"  Prediction shape: {prediction.shape}")
            print(f"  Prediction error: {error:.6f}")
            
            # Check for NaN
            if torch.isnan(prediction).any():
                print(f"  ⚠️  WARNING: NaN detected in prediction!")
            else:
                print(f"  ✓ No NaN values")
            
            errors.append(error)
        else:
            print(f"  ⚠️  Sequence too short for prediction")
    
    # Summary
    if errors:
        print("\n" + "-"*60)
        print("Summary:")
        print(f"  Mean prediction error: {np.mean(errors):.6f}")
        print(f"  Std prediction error: {np.std(errors):.6f}")
        print(f"  Min error: {np.min(errors):.6f}")
        print(f"  Max error: {np.max(errors):.6f}")
    
    print("\n✓ Inference tests complete!")
    print("="*60)
    
    return errors


def test_future_prediction(model):
    """Test multi-step future prediction"""
    print("\n" + "="*60)
    print("Testing Multi-Step Future Prediction")
    print("="*60)
    
    # Create test sample
    dataset = SyntheticHandwritingDataset(num_lines=1, seed=123)
    img, text = dataset[0]
    
    print(f"Test text: '{text}'")
    
    # Extract columns
    columns = extract_columns(
        img,
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE
    ).unsqueeze(0)  # (1, T, H, W)
    
    with torch.no_grad():
        # Encode
        z_seq = model.encode_sequence(columns)
        
        # Predict 5 future steps
        z_future = model.predict_future(columns, steps=5)
    
    print(f"Input embeddings shape: {z_seq.shape}")
    print(f"Future predictions shape: {z_future.shape}")
    
    # Check for NaN
    if torch.isnan(z_future).any():
        print(f"⚠️  WARNING: NaN detected in future predictions!")
    else:
        print(f"✓ All future predictions are valid (no NaN)")
    
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HWM Inference')
    parser.add_argument('--model', type=str, default='hwm_model.pt', help='Model checkpoint')
    parser.add_argument('--num-tests', type=int, default=5, help='Number of inference tests')
    
    args = parser.parse_args()
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load model
    model = load_model(args.model)
    
    # Run tests
    run_inference_tests(model, num_tests=args.num_tests)
    
    # Test future prediction
    test_future_prediction(model)
