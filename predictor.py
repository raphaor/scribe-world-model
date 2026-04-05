"""
HWM Predictor - Transformer for next-step prediction
Predicts future embeddings from past embeddings
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Lightweight transformer for predicting next embedding
    Input: sequence of embeddings (B, T, D)
    Output: predicted next embedding (B, D)
    """
    
    def __init__(
        self,
        embedding_dim=64,
        num_layers=2,
        num_heads=2,
        ff_dim=128,
        dropout=0.1,
        max_seq_len=5000
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            embedding_dim, 
            max_len=max_seq_len, 
            dropout=dropout
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, z_seq):
        """
        Args:
            z_seq: (B, T, D) sequence of embeddings
        Returns:
            z_pred: (B, D) predicted next embedding
        """
        # Add positional encoding
        z_seq = self.pos_encoder(z_seq)
        
        # Transformer encoding
        # Note: using causal mask would be better but keeping simple
        encoded = self.transformer(z_seq)
        
        # Take last timestep and project
        z_last = encoded[:, -1, :]  # (B, D)
        z_pred = self.output_proj(z_last)
        
        return z_pred
    
    def predict_sequence(self, z_seq, steps=1):
        """
        Predict multiple future steps autoregressively
        Args:
            z_seq: (B, T, D)
            steps: number of future steps to predict
        Returns:
            predictions: (B, steps, D)
        """
        predictions = []
        current_seq = z_seq
        
        for _ in range(steps):
            z_pred = self.forward(current_seq)  # (B, D)
            predictions.append(z_pred)
            
            # Append prediction to sequence
            z_pred = z_pred.unsqueeze(1)  # (B, 1, D)
            current_seq = torch.cat([current_seq, z_pred], dim=1)
        
        return torch.stack(predictions, dim=1)  # (B, steps, D)


def test_predictor():
    """Test predictor on Pi"""
    print("\nTesting TransformerPredictor...")
    
    predictor = TransformerPredictor(
        embedding_dim=64,
        num_layers=2,
        num_heads=2,
        ff_dim=128,
        dropout=0.1
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in predictor.parameters())
    print(f"Predictor parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    z_seq = torch.randn(batch_size, seq_len, 64)
    
    with torch.no_grad():
        z_pred = predictor(z_seq)
        z_multi = predictor.predict_sequence(z_seq, steps=3)
    
    print(f"Input shape: {z_seq.shape}")
    print(f"Single prediction shape: {z_pred.shape}")
    print(f"Multi-step prediction shape: {z_multi.shape}")
    print(f"✓ Predictor working!")
    
    return predictor


if __name__ == "__main__":
    test_predictor()
