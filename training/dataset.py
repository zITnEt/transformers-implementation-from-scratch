import os
import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import Tokenizer, PAD_ID
from config import config

RAW_PATH = "data/raw/rus.txt"
CACHE_PATH = "data/processed/pairs.pt"


def build_pairs():
    """Encode every (english, russian) line of the raw corpus into ID pairs.

    The encoded pairs are cached to disk so subsequent runs skip tokenization.
    """
    if os.path.exists(CACHE_PATH):
        return torch.load(CACHE_PATH)

    src_tok = Tokenizer("tokenizer/vocab_en.json", "english")
    tgt_tok = Tokenizer("tokenizer/vocab_ru.json", "russian")

    pairs = []
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            ar = line.strip().split("\t")
            if len(ar) < 2:
                continue
            src_ids = src_tok.encode(ar[0])   # [SOS, ..., EOS]
            tgt_ids = tgt_tok.encode(ar[1])

            # Drop pairs longer than the positional table.
            if len(src_ids) > config.max_seq_len or len(tgt_ids) > config.max_seq_len:
                continue
            pairs.append((src_ids, tgt_ids))

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    torch.save(pairs, CACHE_PATH)
    return pairs


class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch):
    """Pad each side of the batch to its longest sequence.

    Returns src [batch, max_src_len] and tgt [batch, max_tgt_len].
    """
    src_seqs, tgt_seqs = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD_ID)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_ID)
    return src, tgt
