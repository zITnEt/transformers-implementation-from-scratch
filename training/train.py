import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import config
from model.transformer import Transformer, create_src_mask, create_tgt_mask
from tokenizer.tokenizer import Tokenizer, PAD_ID
from training.dataset import build_pairs, TranslationDataset, collate_fn

CHECKPOINT_DIR = "checkpoints"


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, total_batches = 0.0, 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        # Teacher forcing: decoder input is the target shifted right, labels
        # are the target shifted left.
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        src_mask = create_src_mask(src, PAD_ID)
        tgt_mask = create_tgt_mask(tgt_input, PAD_ID)

        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Flatten to [batch*tgt_len, vocab] / [batch*tgt_len] for the loss.
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_labels.reshape(-1),
        )

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    src_tok = Tokenizer("tokenizer/vocab_en.json", "english")
    tgt_tok = Tokenizer("tokenizer/vocab_ru.json", "russian")

    print("loading/encoding data (first run builds the cache)...")
    pairs = build_pairs()
    dataset = TranslationDataset(pairs)
    print(f"{len(dataset)} sentence pairs")

    # Hold out a 1% validation split.
    val_size = max(int(0.01 * len(dataset)), 1)
    train_set, val_set = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # num_workers: load/collate batches in parallel CPU processes so the GPU
    # isn't left waiting between steps. pin_memory speeds the host->GPU copy.
    # (On Windows these workers re-import this module, which is why the
    # entry point is guarded by `if __name__ == "__main__"`.)
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True,
                            persistent_workers=True)

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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params/1e6:.1f}M parameters")

    # Ignore padding in the loss; label smoothing 0.1 as in the paper.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.98), eps=1e-9)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        mins = (time.time() - t0) / 60

        print(f"epoch {epoch:2d} | train {train_loss:.3f} | val {val_loss:.3f} | {mins:.1f} min")

        # Save the latest checkpoint, and the best-on-validation separately.
        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  -> new best (val {val_loss:.3f}), saved checkpoints/best.pt")


if __name__ == "__main__":
    main()
