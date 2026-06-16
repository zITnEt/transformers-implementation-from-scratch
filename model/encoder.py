import torch.nn as nn
from .attention import Attention
from .feedforward import FeedForwardNetwork


class EncoderLayer(nn.Module):
    """Pre-norm encoder block: self-attention + feed-forward, each residual."""

    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = Attention(d_model, heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
    
class Encoder(nn.Module):
    """Stack of N encoder layers with a final LayerNorm (pre-norm convention)."""

    def __init__ (self, d_model, heads, d_ff, N, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout)
                                     for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
