import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"

        self.heads = heads
        self.d_model = d_model
        self.head_dim = d_model // heads

        # Fused projections: a single [d_model, d_model] matmul is equivalent to
        # `heads` separate per-head projections, recovered by reshaping the output.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # [batch, seq, d_model] -> [batch, heads, seq, head_dim]
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        # query/key/value: [batch, seq, d_model]. Equal for self-attention;
        # for cross-attention query is the decoder state, key/value the encoder.
        batch_size = query.size(0)

        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # Scaled dot-product scores: [batch, heads, q_len, k_len].
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Mask out padding / future positions before softmax.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)

        x = torch.matmul(A, V)  # [batch, heads, seq, head_dim]

        # Concatenate heads: [batch, heads, seq, head_dim] -> [batch, seq, d_model].
        x = x.transpose(1, 2).contiguous().reshape(batch_size, -1, self.d_model)

        return self.W_o(x)
