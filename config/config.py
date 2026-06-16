# ---------------- model ----------------

d_model = 128            # model / embedding dimension
max_seq_len = 128        # maximum sequence length (bounds the positional table)
dropout = 0.1

num_heads = 8            # must divide d_model (head dim = 128 / 8 = 16)

num_encoder_layers = 6
num_decoder_layers = 6

d_ff = 512               # feed-forward inner dimension (4 * d_model)

# ---------------- training ----------------

# Sentence pairs per gradient step. The output logits tensor is
# batch * seq * vocab, and the ~62k Russian vocab dominates GPU memory:
# 128 fits a 16 GB GPU, use 32 on an 8 GB GPU.
batch_size = 128

learning_rate = 3e-4     # flat Adam LR (no warmup schedule)

num_epochs = 10
