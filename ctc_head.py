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
