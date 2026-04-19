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
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        """
        x = x + self.pe[:, : x.size(1)]
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
        max_seq_len=5000,
        causal=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.causal = causal

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            embedding_dim, max_len=max_seq_len, dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def _generate_causal_mask(self, seq_len, device):
        """Upper triangular mask: True = masked (cannot attend)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

    def forward(self, z_seq, src_key_padding_mask=None):
        """
        Args:
            z_seq: (B, T, D) sequence of embeddings
            src_key_padding_mask: (B, T) bool — True = position to ignore
                as a key. Used by the v7 I-JEPA context-encoder role so
                that target/padding positions don't contribute to the
                context representation.
        Returns:
            z_pred: (B, T, D) predicted embedding at each position.
            In causal mode: predicts the next step given the past.
            In non-causal (JEPA) mode: bidirectional attention over the
            whole sequence — used to fill in masked positions.
        """
        z_seq = self.pos_encoder(z_seq)

        if self.causal:
            causal_mask = self._generate_causal_mask(z_seq.size(1), z_seq.device)
            encoded = self.transformer(
                z_seq, mask=causal_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        else:
            encoded = self.transformer(
                z_seq, src_key_padding_mask=src_key_padding_mask
            )

        z_pred = self.output_proj(encoded)  # (B, T, D)

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
            z_all = self.forward(current_seq)  # (B, T, D)
            z_next = z_all[:, -1, :]           # (B, D)
            predictions.append(z_next)
            current_seq = torch.cat([current_seq, z_next.unsqueeze(1)], dim=1)

        return torch.stack(predictions, dim=1)  # (B, steps, D)


class JEPACrossAttnPredictor(nn.Module):
    """
    I-JEPA style predictor with cross-attention to a context encoder.

    The v5/v6 predictor does in-place mask-token filling: mask tokens
    are substituted in the input and self-attention fills them. That
    lets context positions see mask tokens, which can leak subtle
    information and blurs the "predict from context" boundary.

    This predictor separates the two cleanly:

      1. A *context encoder* (self-attention, runs outside this module)
         processes z_seq with src_key_padding_mask hiding target and
         padding positions, producing z_ctx.
      2. This predictor builds *queries* at every sequence position
         (mask_token + positional encoding) and runs a stack of
         TransformerDecoder layers that cross-attend to z_ctx while
         self-attending among queries.
      3. The caller reads the output at masked positions to get the
         predictions.

    Queries never leak into the context representation, and targets
    are genuinely predicted from context alone.
    """

    def __init__(
        self,
        embedding_dim,
        num_layers,
        num_heads,
        ff_dim,
        dropout=0.1,
        max_seq_len=5000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_encoder = PositionalEncoding(
            embedding_dim, max_len=max_seq_len, dropout=dropout
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, context, memory_key_padding_mask, mask_token, seq_len):
        """
        Args:
            context: (B, T, D) — context encoder output (all positions).
            memory_key_padding_mask: (B, T) bool — True = ignore this
                context position in cross-attention (targets + padding).
            mask_token: (1, 1, D) — learnable [MASK] vector, expanded
                to build queries.
            seq_len: T — query sequence length (== context.size(1)).
        Returns:
            pred: (B, T, D) — predictions at every position; caller
                selects the rows at masked positions.
        """
        B, T, D = context.size(0), seq_len, context.size(2)
        # Queries = mask_token at every position + positional encoding.
        # We build queries at all T positions (not only masked) so the
        # caller can index with the same boolean mask used elsewhere.
        queries = mask_token.expand(B, T, D).contiguous()
        queries = self.pos_encoder(queries)
        out = self.decoder(
            tgt=queries,
            memory=context,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)


def test_predictor():
    """Test predictor on Pi"""
    print("\nTesting TransformerPredictor...")

    predictor = TransformerPredictor(
        embedding_dim=64, num_layers=2, num_heads=2, ff_dim=128, dropout=0.1
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
