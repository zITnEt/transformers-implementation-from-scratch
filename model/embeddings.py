import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Token embedding scaled by sqrt(d_model), as in the original Transformer."""

    def __init__(self, vocab_size, d_model, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        # [batch, seq] (token IDs) -> [batch, seq, d_model]
        return self.embedding(x) * math.sqrt(self.d_model)
