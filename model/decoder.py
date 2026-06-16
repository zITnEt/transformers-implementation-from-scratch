import torch.nn as nn
from .attention import Attention
from .feedforward import FeedForwardNetwork


class DecoderLayer(nn.Module):
    """Pre-norm decoder block: masked self-attention, cross-attention, feed-forward.

    forward(x_enc, x_dec, mask1, mask2): mask1 is the causal target mask for
    self-attention, mask2 the source padding mask for cross-attention.
    """

    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = Attention(d_model, heads, dropout)
        self.cross_attn = Attention(d_model, heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, x_dec, mask1=None, mask2=None):
        x_norm = self.norm1(x_dec)
        x = x_dec + self.dropout(self.self_attn(x_norm, x_norm, x_norm, mask1))
        x_norm = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x_norm, x_enc, x_enc, mask2))
        x_norm = self.norm3(x)
        x = x + self.dropout(self.ffn(x_norm))
        return x
    
class Decoder(nn.Module):
    """Stack of N decoder layers with a final LayerNorm (pre-norm convention)."""

    def __init__(self, d_model, heads, d_ff, N, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) 
                                     for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)


    def forward(self, x_enc, x, mask1=None, mask2=None):
        for layer in self.layers:
            x = layer(x_enc, x, mask1, mask2)
        
        return self.norm(x)