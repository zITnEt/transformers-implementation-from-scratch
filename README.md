# Transformer NMT — English → Russian (from scratch)

A sequence-to-sequence **Transformer** built from first principles in PyTorch,
following *Attention Is All You Need* (Vaswani et al., 2017), in the pre-norm
variant. Multi-head attention, positional encoding, and the encoder–decoder
stack are all hand-implemented — no `transformers`, no pre-built attention
layers, no pre-trained weights.

Trained end-to-end on **536k** Tatoeba English–Russian sentence pairs.

```
$ python -m inference.translate "Black cat is on the table."
на столе чёрная кошка
```

## Architecture

| Component | Implementation |
|---|---|
| Attention | Multi-head scaled dot-product, fused QKV projections (`model/attention.py`) |
| Positional encoding | Fixed sinusoidal (`model/positional.py`) |
| Encoder / decoder | Pre-norm residual blocks, 6 layers each (`model/encoder.py`, `model/decoder.py`) |
| Feed-forward | Position-wise, GELU (`model/feedforward.py`) |
| Tokenizer | Word-level, NLTK (`tokenizer/`) |
| Training | Teacher forcing, label smoothing, gradient clipping (`training/train.py`) |
| Inference | Greedy autoregressive decoding (`inference/translate.py`) |

| Hyperparameter | Value |
|---|---|
| `d_model` | 128 |
| Attention heads | 8 |
| Encoder / decoder layers | 6 / 6 |
| Feed-forward dim | 512 |
| Parameters | 21.2M |
| Vocab (EN / RU) | 18,719 / 62,446 |
| Optimizer | Adam, lr 3e-4, label smoothing 0.1 |

## Project layout

```
config/config.py     hyperparameters
model/               attention, feed-forward, embeddings, positional encoding,
                     encoder/decoder stacks, full Transformer + mask helpers
tokenizer/           vocab builder + word-level tokenizer (encode/decode)
training/            corpus encoding/caching, Dataset, collate, training loop
inference/           greedy decoding (CLI + interactive)
data/raw/rus.txt     Tatoeba EN<TAB>RU pairs
```

## Setup

```bash
pip install -r requirements.txt
python setup.py                 # download NLTK tokenizer data
python tokenizer/vocab.py       # build vocab jsons (only if missing)
```

## Train

```bash
python -m training.train
```

The corpus is tokenized once and cached to `data/processed/pairs.pt`; later runs
load it instantly. Checkpoints are written to `checkpoints/last.pt` (every epoch)
and `checkpoints/best.pt` (best validation loss). Tune `batch_size`,
`learning_rate`, and `num_epochs` in `config/config.py`.

## Translate

```bash
python -m inference.translate "I want to drink a cold beer."   # one-shot
python -m inference.translate                                  # interactive
```

Greedy decoding from `checkpoints/best.pt`.

## Scope & limitations

This is a compact model trained as a from-scratch implementation study. It
translates everyday sentences within the Tatoeba domain; it is not intended to
match production systems. Known limitations follow directly from the design
choices: the **word-level tokenizer** maps unseen or misspelled words to `<unk>`,
and **greedy decoding** can repeat tokens on harder inputs. Natural next steps
would be subword (BPE) tokenization, beam search, and a larger model.
