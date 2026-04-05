"""
HWM Encoder - CNN 1D for encoding handwriting columns
Converts image columns to latent embeddings
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Lightweight CNN 1D encoder for handwriting columns
    Input: (B, H) or (B, H, W) where W is window_size
    Output: (B, embedding_dim)
    """

    def __init__(self, img_height=32, window_size=10, embedding_dim=64):
        super().__init__()

        self.img_height = img_height
        self.window_size = window_size
        input_dim = img_height * window_size

        # Very lightweight: 2 conv layers + 1 linear
        self.net = nn.Sequential(
            # Input: (B, H*W)
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim),
        )

        # Alternative: Conv1D approach
        self.use_conv = False
        if self.use_conv:
            # Treat as 1D sequence: (B, 1, H*W)
            self.conv_net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, embedding_dim),
            )

    def forward(self, x):
        """
        Args:
            x: (B, H, W) or (B, H*W) tensor
        Returns:
            z: (B, embedding_dim) embeddings
        """
        # Flatten if needed
        if x.dim() == 3:
            B, H, W = x.shape
            x = x.view(B, -1)

        if self.use_conv:
            # Add channel dimension for Conv1D
            x = x.unsqueeze(1)
            return self.conv_net(x)
        else:
            return self.net(x)


class Conv2DEncoder(nn.Module):
    """
    Conv2D encoder for handwriting image columns.
    Preserves spatial structure unlike the MLP encoder.

    Input: (B, H, W) where H=48, W=window_size
    Output: (B, embedding_dim)
    """

    def __init__(self, img_height=48, window_size=10, embedding_dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
        )
        self.fc = nn.Linear(128 * 4, embedding_dim)

    def forward(self, x):
        if x.dim() == 2:
            raise ValueError("Conv2DEncoder requires (B, H, W) input")
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_encoder():
    """Test encoder on Pi"""
    print("Testing CNNEncoder...")

    encoder = CNNEncoder(img_height=32, window_size=10, embedding_dim=64)

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 32, 10)

    with torch.no_grad():
        z = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")
    print(f"✓ Encoder working!")

    return encoder


if __name__ == "__main__":
    test_encoder()
