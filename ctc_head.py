"""
CTC Recognition Head - maps embeddings to character logits.
"""

import torch.nn as nn


class CTCHead(nn.Module):
    """
    Linear projection from embedding space to character probabilities.

    Input: (B, T, D) embedding sequence
    Output: (B, T, num_classes) log-probabilities
    """

    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, num_classes)

    def forward(self, z_seq):
        return self.proj(z_seq).log_softmax(dim=-1)


class CTCHeadBiLSTM(nn.Module):
    """
    CTC head with BiLSTM context layer.
    Each frame sees its neighbors before making a character decision.

    Input: (B, T, D) embedding sequence
    Output: (B, T, num_classes) log-probabilities
    """

    def __init__(self, embedding_dim, num_classes, hidden_dim=256, num_lstm_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=0.1 if num_lstm_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, z_seq):
        ctx, _ = self.lstm(z_seq)  # (B, T, hidden*2)
        return self.proj(ctx).log_softmax(dim=-1)
