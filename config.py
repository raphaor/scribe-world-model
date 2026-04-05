"""
HWM-v1 Configuration
Pi-friendly hyperparameters for lightweight world model
"""

# Architecture
EMBEDDING_DIM = 64          # Latent dimension (vs 192 in LeWM)
NUM_LAYERS = 2              # Transformer layers
NUM_HEADS = 2               # Attention heads
FF_DIM = 128                # Feed-forward dimension
DROPOUT = 0.1

# Input processing
WINDOW_SIZE = 10            # Columns per "frame" (sliding window)
STRIDE = 5                  # Stride for sliding window
IMG_HEIGHT = 32             # Standardized line height

# Training
BATCH_SIZE = 4              # Small for Pi memory
SEQ_LEN = 50                # Short sequences for testing
LEARNING_RATE = 1e-3
SIGREG_LAMBDA = 0.1         # Regularization strength
MAX_PARAMS = 1_000_000      # Hard limit for Pi

# Data
NUM_SYNTHETIC_LINES = 100   # Small dataset for PoC
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Limited alphabet

# Logging
LOG_INTERVAL = 5            # Steps between logs

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
