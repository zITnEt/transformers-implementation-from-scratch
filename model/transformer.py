import torch
import torch.nn as nn
from .embeddings import TokenEmbedding
from .positional import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder


def create_src_mask(src, pad_idx=0):
    """Source padding mask, [batch, 1, 1, src_len], broadcasting over heads and
    query positions. Used by encoder self-attention and decoder cross-attention."""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def create_tgt_mask(tgt, pad_idx=0):
    """Decoder self-attention mask, [batch, 1, tgt_len, tgt_len]: padding mask
    AND causal (lower-triangular) mask so position i attends only to positions <= i."""
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    causal = torch.tril(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device)
    )
    return pad_mask & causal


class Transformer(nn.Module):
    """Encoder-decoder Transformer for sequence-to-sequence translation."""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, heads, d_ff,
                 num_encoder_layers, num_decoder_layers, max_seq_len,
                 dropout=0.1, pad_idx=0):
        super().__init__()

        # Separate source/target embeddings (distinct vocabularies, no weight tying).
        self.src_embed = TokenEmbedding(src_vocab_size, d_model, pad_idx)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model, pad_idx)

        # Shared positional encoding (parameter-free, identical for both sides).
        self.pos = PositionalEncoding(d_model, max_seq_len, dropout)

        self.encoder = Encoder(d_model, heads, d_ff, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, heads, d_ff, num_decoder_layers, dropout)

        # Output projection to vocab logits; softmax is left to the loss / decoder.
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        # [batch, src_len] -> memory [batch, src_len, d_model]
        x = self.pos(self.src_embed(src))
        return self.encoder(x, src_mask)

    def decode(self, memory, tgt, src_mask=None, tgt_mask=None):
        # Self-attention uses the causal target mask; cross-attention uses the
        # source padding mask.
        x = self.pos(self.tgt_embed(tgt))
        return self.decoder(memory, x, tgt_mask, src_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Returns logits [batch, tgt_len, tgt_vocab_size].
        memory = self.encode(src, src_mask)
        out = self.decode(memory, tgt, src_mask, tgt_mask)
        return self.output_proj(out)
