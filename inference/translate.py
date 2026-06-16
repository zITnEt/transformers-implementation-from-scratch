import sys
import torch

from config import config
from model.transformer import Transformer, create_src_mask, create_tgt_mask
from tokenizer.tokenizer import Tokenizer, SOS_ID, EOS_ID, PAD_ID

CHECKPOINT = "checkpoints/best.pt"


def load_model(device):
    src_tok = Tokenizer("tokenizer/vocab_en.json", "english")
    tgt_tok = Tokenizer("tokenizer/vocab_ru.json", "russian")

    model = Transformer(
        src_vocab_size=len(src_tok),
        tgt_vocab_size=len(tgt_tok),
        d_model=config.d_model,
        heads=config.num_heads,
        d_ff=config.d_ff,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        pad_idx=PAD_ID,
    ).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, src_tok, tgt_tok


@torch.no_grad()
def greedy_translate(model, src_tok, tgt_tok, sentence, device):
    # Encode the source once; the memory is reused for every decoding step.
    src = torch.tensor([src_tok.encode(sentence)], device=device)  # [1, src_len]
    src_mask = create_src_mask(src, PAD_ID)
    memory = model.encode(src, src_mask)

    # Autoregressive decode: start from <sos> and append one token per step.
    tgt = torch.tensor([[SOS_ID]], device=device)                  # [1, 1]

    for _ in range(config.max_seq_len - 1):
        tgt_mask = create_tgt_mask(tgt, PAD_ID)
        out = model.decode(memory, tgt, src_mask, tgt_mask)
        logits = model.output_proj(out[:, -1])     # last position only
        next_id = logits.argmax(dim=-1)            # greedy selection
        tgt = torch.cat([tgt, next_id.unsqueeze(0)], dim=1)
        if next_id.item() == EOS_ID:
            break

    return tgt_tok.decode(tgt[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src_tok, tgt_tok = load_model(device)

    if len(sys.argv) > 1:
        # One-shot: python -m inference.translate "I am hungry."
        print(greedy_translate(model, src_tok, tgt_tok, " ".join(sys.argv[1:]), device))
        return

    # Interactive mode: read a sentence per line until an empty line.
    print("type an English sentence (empty line to quit)")
    while True:
        try:
            sentence = input("> ").strip()
        except EOFError:
            break
        if not sentence:
            break
        print(greedy_translate(model, src_tok, tgt_tok, sentence, device))


if __name__ == "__main__":
    main()
    