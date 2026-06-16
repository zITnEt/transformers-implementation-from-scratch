# English → Russian Neural Machine Translation — Transformer from Scratch

A sequence-to-sequence **Transformer** (encoder–decoder) implementing the
architecture from *Attention Is All You Need* (Vaswani et al., 2017) — built
from first principles in PyTorch, **without** using `transformers`, pre-built
attention layers, or pre-trained weights.

> **Result:** trains end-to-end on 536,124 English–Russian sentence pairs and
> produces fluent translations of everyday sentences.

![Translation examples](screenshots/translations.png)
*<sup>Replace with your screenshot of `inference.translate` output.</sup>*

---

## What I built from scratch

Every core component is hand-implemented — no high-level NLP libraries:

- **Multi-head scaled dot-product attention** (`model/attention.py`)
- **Sinusoidal positional encoding** (`model/positional.py`)
- **Pre-norm encoder / decoder stacks** with residual connections (`model/encoder.py`, `model/decoder.py`)
- **Position-wise feed-forward networks** (`model/feedforward.py`)
- **Word-level tokenizer + vocab builder** (`tokenizer/`)
- **Training loop** with teacher forcing, label smoothing, gradient clipping, and validation tracking (`training/train.py`)
- **Greedy autoregressive decoding** for inference (`inference/translate.py`)

![Training run](screenshots/training.png)
*<sup>GPU training: 21.2M parameters over 536k pairs.</sup>*

![Loss curve](loss_curve.png)
*<sup>Train and validation loss falling over 10 epochs.</sup>*

---

## Model

| Hyperparameter | Value |
|---|---|
| Embedding dim (`d_model`) | 128 |
| Attention heads | 8 |
| Encoder / decoder layers | 6 / 6 |
| Feed-forward dim | 512 |
| Parameters | 21.2M |
| Vocab (EN / RU) | 18,719 / 62,446 |
| Optimizer | Adam (3e-4), label smoothing 0.1 |
| Data | Tatoeba EN–RU, 536,124 pairs |

## Engineering highlights

- Designed the full **encoder–decoder data flow** including causal + padding masks.
- Handled **GPU memory constraints**: the 62k-word output vocabulary makes the
  logits tensor the memory bottleneck — tuned batch size to fit the available GPU.
- Cached the tokenized dataset to disk so re-runs start instantly.
- Trained on GPU (CUDA), with per-epoch checkpointing of the best model.

## Run it

```bash
python setup.py                          # one-time: nltk tokenizer data
python -m training.train                 # train (checkpoints to checkpoints/)
python -m inference.translate "I am hungry."   # translate
```

## Tech stack

Python · PyTorch · NLTK · CUDA
