import json
import nltk

# Special token IDs; must match the order used when the vocab jsons were built.
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class Tokenizer:
    """Word-level tokenizer backed by a vocab json.

    encode: text -> [SOS_ID, ..., EOS_ID]; decode: ids -> text (specials stripped).
    """

    def __init__(self, vocab_path, language):
        # language selects the nltk tokenizer model ('english' / 'russian').
        self.language = language

        with open(vocab_path, encoding="utf-8") as f:
            self.stoi = json.load(f)          # token -> id
        self.itos = {i: tok for tok, i in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def tokenize(self, text):
        # Lowercase then split, matching the normalization used to build the vocab.
        return nltk.word_tokenize(text.lower(), language=self.language)

    def encode(self, text, add_special=True):
        ids = [self.stoi.get(tok, UNK_ID) for tok in self.tokenize(text)]
        if add_special:
            ids = [SOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            i = int(i)                         # accept tensors too
            if i == EOS_ID:
                break
            if i in (PAD_ID, SOS_ID):
                continue
            words.append(self.itos.get(i, "<unk>"))
        return " ".join(words)
