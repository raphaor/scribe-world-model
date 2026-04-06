"""
HWM-v1 Model - Handwriting World Model
Complete architecture combining encoder and predictor
"""

import torch
import torch.nn as nn

from encoder import CNNEncoder, Conv2DEncoder, Conv2DEncoderV2
from predictor import TransformerPredictor
from loss import HWMLoss, HybridLoss
from ctc_head import CTCHead
import config


class HWMv1(nn.Module):
    """
    Handwriting World Model v1

    Architecture:
        Image columns → [Encoder] → Embeddings → [Predictor] → Next embedding

    Training objective:
        Predict z_{t+1} from z_{0:t}
        Loss: MSE(z_{t+1}, ż_{t+1}) + λ * SIGReg(Z)
    """

    def __init__(
        self,
        img_height=32,
        window_size=10,
        embedding_dim=64,
        num_layers=2,
        num_heads=2,
        ff_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.img_height = img_height
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        # Encoder: columns → embeddings
        self.encoder = CNNEncoder(
            img_height=img_height, window_size=window_size, embedding_dim=embedding_dim
        )

        # Predictor: embedding sequence → next embedding
        self.predictor = TransformerPredictor(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # Loss function
        self.criterion = HWMLoss(lambda_sigreg=config.SIGREG_LAMBDA)

    def encode_sequence(self, img_columns):
        """
        Encode image columns to embeddings
        Args:
            img_columns: (B, T, H, W) or (B, T, H*W)
        Returns:
            z_seq: (B, T, D) embeddings
        """
        B, T = img_columns.shape[:2]

        # Flatten batch and time
        if img_columns.dim() == 4:
            H, W = img_columns.shape[2], img_columns.shape[3]
            img_columns = img_columns.view(B, T, H, W)
            img_flat = img_columns.view(B * T, H, W)
        else:
            img_flat = img_columns.view(B * T, -1)

        # Encode
        z_flat = self.encoder(img_flat)  # (B*T, D)

        # Reshape to sequence
        z_seq = z_flat.view(B, T, -1)  # (B, T, D)

        return z_seq

    def forward(self, img_columns):
        """
        Forward pass: encode columns and predict next embeddings (dense)
        Args:
            img_columns: (B, T, H, W) where T >= 2
        Returns:
            z_pred: (B, T-1, D) predicted next embedding at each position
            z_seq: (B, T, D) all embeddings
        """
        z_seq = self.encode_sequence(img_columns)       # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])       # (B, T-1, D)
        return z_pred, z_seq

    def compute_loss(self, img_columns):
        """
        Compute training loss
        Args:
            img_columns: (B, T, H, W) where T >= 2
        Returns:
            total_loss, losses_dict
        """
        z_pred, z_seq = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        total_loss, losses_dict = self.criterion(z_pred, z_target, z_seq)
        return total_loss, losses_dict

    def predict_future(self, img_columns, steps=1):
        """
        Predict multiple future embeddings
        Args:
            img_columns: (B, T, H, W)
            steps: number of future steps
        Returns:
            z_future: (B, steps, D)
        """
        # Encode
        z_seq = self.encode_sequence(img_columns)

        # Predict
        z_future = self.predictor.predict_sequence(z_seq, steps=steps)

        return z_future

    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv2(nn.Module):
    """
    Handwriting World Model v2

    Architecture:
        Image columns -> [Conv2D Encoder] -> Embeddings
                                               |
                              +----------------+----------------+
                              |                                 |
                   [Transformer Predictor]          [CTC Recognition Head]
                   (next embedding, causal)         (character logits)
    """

    def __init__(
        self,
        img_height=48,
        window_size=10,
        embedding_dim=96,
        num_layers=2,
        num_heads=2,
        ff_dim=192,
        dropout=0.1,
        num_classes=None,
        lambda_sigreg=0.1,
        lambda_ctc=1.0,
    ):
        super().__init__()

        self.img_height = img_height
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        self.encoder = Conv2DEncoder(img_height, window_size, embedding_dim)
        self.predictor = TransformerPredictor(
            embedding_dim, num_layers, num_heads, ff_dim, dropout
        )

        self.ctc_head = CTCHead(embedding_dim, num_classes) if num_classes else None
        self.criterion = HybridLoss(lambda_sigreg, lambda_ctc)

    def encode_sequence(self, img_columns):
        B, T = img_columns.shape[:2]
        z_seq = self.encoder(
            img_columns.reshape(B * T, img_columns.shape[2], img_columns.shape[3])
        ).view(B, T, -1)
        return z_seq

    def forward(self, img_columns):
        z_seq = self.encode_sequence(img_columns)       # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])       # (B, T-1, D)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head else None
        return z_pred, z_seq, ctc_logits

    def compute_loss(
        self, img_columns, targets=None, input_lengths=None, target_lengths=None
    ):
        z_pred, z_seq, ctc_logits = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        return self.criterion(
            z_pred, z_target, z_seq, ctc_logits, targets, input_lengths, target_lengths
        )

    def adapt(self, img_columns):
        z_pred, z_seq, _ = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        return self.criterion(z_pred, z_target, z_seq)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HWMv3(nn.Module):
    """
    Handwriting World Model v3
    Deeper encoder, wider window, larger transformer.
    """

    def __init__(
        self,
        img_height=48,
        window_size=32,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=384,
        dropout=0.1,
        num_classes=None,
        lambda_sigreg=0.1,
        lambda_ctc=2.0,
    ):
        super().__init__()
        self.img_height = img_height
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        self.encoder = Conv2DEncoderV2(img_height, window_size, embedding_dim)
        self.predictor = TransformerPredictor(
            embedding_dim, num_layers, num_heads, ff_dim, dropout
        )
        self.ctc_head = CTCHead(embedding_dim, num_classes) if num_classes else None
        self.criterion = HybridLoss(lambda_sigreg, lambda_ctc)

    def encode_sequence(self, img_columns):
        B, T = img_columns.shape[:2]
        z_seq = self.encoder(
            img_columns.reshape(B * T, img_columns.shape[2], img_columns.shape[3])
        ).view(B, T, -1)
        return z_seq

    def forward(self, img_columns):
        z_seq = self.encode_sequence(img_columns)       # (B, T, D)
        z_pred = self.predictor(z_seq[:, :-1, :])       # (B, T-1, D)
        ctc_logits = self.ctc_head(z_seq) if self.ctc_head else None
        return z_pred, z_seq, ctc_logits

    def compute_loss(
        self, img_columns, targets=None, input_lengths=None, target_lengths=None
    ):
        z_pred, z_seq, ctc_logits = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        return self.criterion(
            z_pred, z_target, z_seq, ctc_logits, targets, input_lengths, target_lengths
        )

    def adapt(self, img_columns):
        z_pred, z_seq, _ = self.forward(img_columns)
        z_target = z_seq[:, 1:, :].detach()             # (B, T-1, D)
        return self.criterion(z_pred, z_target, z_seq)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model():
    """Create HWM model with config defaults"""
    model = HWMv1(
        img_height=config.IMG_HEIGHT,
        window_size=config.WINDOW_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM,
        dropout=config.DROPOUT,
    )
    return model


def test_model():
    """Test complete model"""
    print("\n" + "=" * 60)
    print("Testing HWM-v1 Model")
    print("=" * 60)

    # Create model
    model = create_model()

    # Count parameters
    total_params = model.count_parameters()
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    predictor_params = sum(p.numel() for p in model.predictor.parameters())

    print(f"\nParameter counts:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Predictor: {predictor_params:,}")
    print(f"  Total: {total_params:,}")

    if total_params > config.MAX_PARAMS:
        print(f"  ⚠️  WARNING: Exceeds max params ({config.MAX_PARAMS:,})")
    else:
        print(f"  ✓ Under max params limit ({config.MAX_PARAMS:,})")

    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = config.BATCH_SIZE
    seq_len = 10
    img_columns = torch.randn(
        batch_size, seq_len, config.IMG_HEIGHT, config.WINDOW_SIZE
    )

    with torch.no_grad():
        z_pred, z_seq = model(img_columns)
        z_future = model.predict_future(img_columns, steps=3)

    print(f"  Input shape: {img_columns.shape}")
    print(f"  Embeddings shape: {z_seq.shape}")
    print(f"  Prediction shape: {z_pred.shape}")
    print(f"  Future predictions shape: {z_future.shape}")

    # Test loss computation
    print(f"\nTesting loss computation...")
    total_loss, losses_dict = model.compute_loss(img_columns)
    print(f"  Total loss: {losses_dict['total']:.4f}")
    print(f"  Pred loss: {losses_dict['pred']:.4f}")
    print(f"  SIGReg loss: {losses_dict['sigreg']:.4f}")

    print(f"\n✓ Model working correctly!")
    print("=" * 60)

    return model


if __name__ == "__main__":
    test_model()
