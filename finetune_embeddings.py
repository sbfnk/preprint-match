#!/usr/bin/env python3
"""
Fine-tune SPECTER2 embeddings via contrastive learning for journal prediction.

Uses InfoNCE loss with in-batch negatives: papers from the same journal are
positive pairs, other papers in the batch are negatives. Only adapter weights
are fine-tuned (~0.9M params), keeping the base model frozen to reduce
catastrophic forgetting.

After training, regenerates all embeddings using the fine-tuned model and
saves them in the same format as generate_embeddings.py for evaluation with
evaluate_knn.py.

Usage:
  # Train on GPU
  python3 finetune_embeddings.py --output-dir finetuned-specter2/

  # Print SLURM sbatch template
  python3 finetune_embeddings.py --print-sbatch

  # Resume from checkpoint
  python3 finetune_embeddings.py --output-dir finetuned-specter2/ --resume
"""

import json
import argparse
import sys
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from generate_embeddings import load_dataset, select_device
from evaluate_knn import stratified_split, stratified_split_3way


SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=medrxiv-finetune
#SBATCH --output=medrxiv-finetune_%j.out
#SBATCH --error=medrxiv-finetune_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

cd ~/medrxiv
conda activate medrxiv

python3 finetune_embeddings.py \\
    --output-dir finetuned-specter2/ \\
    --epochs 3 \\
    --batch-size 16 \\
    --lr 2e-5 \\
    --temperature 0.05
"""


class PairDataset(Dataset):
    """Dataset of (anchor_record, positive_record) pairs sharing a journal."""

    def __init__(self, records, train_idx, seed=42):
        self.records = records
        self.pairs = self._build_pairs(train_idx, seed)

    def _build_pairs(self, train_idx, seed):
        rng = np.random.default_rng(seed)

        # Group training indices by journal
        journal_groups = defaultdict(list)
        for idx in train_idx:
            journal_groups[self.records[idx]["journal"]].append(idx)

        # For each journal with >= 2 papers, create pairs
        pairs = []
        for journal, indices in journal_groups.items():
            if len(indices) < 2:
                continue
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            # Create pairs: (i, i+1), wrapping around
            for i in range(len(shuffled)):
                anchor = shuffled[i]
                positive = shuffled[(i + 1) % len(shuffled)]
                pairs.append((anchor, positive))

        rng.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_idx, positive_idx = self.pairs[idx]
        return anchor_idx, positive_idx


class HardNegativeBatchSampler:
    """Batch sampler that groups pairs by category for harder in-batch negatives.

    Papers from the same medRxiv category but different journals are harder
    negatives than random papers. By constructing batches within categories,
    each in-batch negative is topically similar to the anchor, forcing the
    model to learn finer-grained journal distinctions.
    """

    def __init__(self, dataset, records, batch_size, seed=42):
        self.batch_size = batch_size
        rng = np.random.default_rng(seed)

        # Group pair indices by anchor's category
        cat_groups = defaultdict(list)
        for i, (anchor_idx, _) in enumerate(dataset.pairs):
            cat = records[anchor_idx].get("category", "")
            cat_groups[cat].append(i)

        # Build batches: draw from same category where possible
        self.batches = []
        for cat, indices in cat_groups.items():
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                if len(batch) > 1:  # skip singleton batches
                    self.batches.append(batch)

        rng.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def embed_paper(record, tokenizer, model, device, stride=256, max_chunks=8):
    """Embed a single paper using chunk + mean-pool, with gradients.

    Same approach as generate_fulltext_embeddings() but keeps gradients for
    backpropagation. Limits to max_chunks evenly-spaced chunks to bound memory.
    """
    title = record.get("title", "") or ""
    abstract = record.get("abstract", "") or ""
    full_text = record.get("full_text", "") or ""

    if full_text:
        text = (title + tokenizer.sep_token + abstract
                + tokenizer.sep_token + full_text)
    else:
        text = title + tokenizer.sep_token + abstract

    encoded = tokenizer(
        text,
        return_overflowing_tokens=True,
        max_length=512,
        stride=stride,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    n_chunks = encoded["input_ids"].shape[0]

    # Subsample evenly-spaced chunks if too many
    if n_chunks > max_chunks:
        indices = np.linspace(0, n_chunks - 1, max_chunks, dtype=int)
        encoded = {
            k: v[indices]
            for k, v in encoded.items()
            if k != "overflow_to_sample_mapping"
        }
        n_chunks = max_chunks

    inputs = {
        k: v.to(device)
        for k, v in encoded.items()
        if k != "overflow_to_sample_mapping"
    }

    outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :]  # [n_chunks, dim]
    paper_emb = cls_emb.mean(dim=0)  # [dim]

    return paper_emb


def contrastive_loss(anchor_embs, positive_embs, temperature=0.05):
    """Symmetric InfoNCE / NT-Xent loss with in-batch negatives.

    Args:
        anchor_embs: [B, D] tensor of anchor embeddings.
        positive_embs: [B, D] tensor of positive embeddings.
        temperature: scaling factor for similarity scores.

    Returns:
        Scalar loss (average of both directions).
    """
    # L2-normalise
    anchor_norm = F.normalize(anchor_embs, dim=1)
    positive_norm = F.normalize(positive_embs, dim=1)

    # Similarity matrix: [B, B]
    sim = anchor_norm @ positive_norm.T / temperature

    # Labels: diagonal (each anchor matches its own positive)
    labels = torch.arange(sim.shape[0], device=sim.device)

    # Symmetric loss
    loss_a2p = F.cross_entropy(sim, labels)
    loss_p2a = F.cross_entropy(sim.T, labels)

    return (loss_a2p + loss_p2a) / 2.0


def save_checkpoint(output_dir, model, optimiser, scheduler, epoch, step,
                    total_loss, best_loss):
    """Save training checkpoint."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"

    torch.save({
        "epoch": epoch,
        "step": step,
        "total_loss": total_loss,
        "best_loss": best_loss,
        "optimiser_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, ckpt_path)

    # Save adapter weights separately
    model.save_adapter(str(ckpt_dir / "adapter"), "[PRX]")
    print(f"  Checkpoint saved at epoch {epoch}, step {step}", file=sys.stderr)


def load_checkpoint(output_dir, model, optimiser, scheduler, device):
    """Load training checkpoint if it exists."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_path = ckpt_dir / "latest.pt"

    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Load adapter weights
    adapter_path = ckpt_dir / "adapter"
    if adapter_path.exists():
        model.load_adapter(str(adapter_path), set_active=True)

    optimiser.load_state_dict(ckpt["optimiser_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    print(f"Resumed from checkpoint: epoch {ckpt['epoch']}, step {ckpt['step']}",
          file=sys.stderr)
    return ckpt


def regenerate_embeddings(records, tokenizer, model, device, output_dir,
                          stride=256, batch_size=32):
    """Regenerate all embeddings using the fine-tuned model.

    Uses the same chunk + mean-pool approach as generate_embeddings.py but
    with the fine-tuned adapter weights.
    """
    from generate_embeddings import generate_fulltext_embeddings, _load_checkpoint

    print("\nRegenerating embeddings with fine-tuned model...", file=sys.stderr)
    model.eval()

    emb_dir = Path(output_dir) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if available
    start_idx = 0
    existing_embeddings = None
    ckpt = _load_checkpoint(emb_dir)
    if ckpt is not None:
        existing_embeddings, start_idx = ckpt
        print(f"Resuming embedding generation from record {start_idx}",
              file=sys.stderr)

    emb = generate_fulltext_embeddings(
        records, tokenizer, model, device,
        batch_size=batch_size,
        stride=stride,
        checkpoint_dir=emb_dir,
        checkpoint_every=1000,
        start_idx=start_idx,
        existing_embeddings=existing_embeddings,
    )

    # Save in same format as generate_embeddings.py
    emb_path = emb_dir / "embeddings.npz"
    np.savez_compressed(emb_path, embeddings=emb)

    metadata = {
        "dois": [r["preprint_doi"] for r in records],
        "journals": [r["journal"] for r in records],
        "categories": [r.get("category", "") for r in records],
        "n_records": len(records),
        "n_journals": len(set(r["journal"] for r in records)),
        "embedding_dim": int(emb.shape[1]),
        "model": "specter2-finetuned",
        "mode": "full-text",
    }
    meta_path = emb_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Clean up checkpoint
    ckpt_path = emb_dir / "checkpoint.npz"
    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"Embeddings saved to {emb_path} ({emb.shape})", file=sys.stderr)
    print(f"Metadata saved to {meta_path}", file=sys.stderr)

    return emb


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SPECTER2 via contrastive learning")
    parser.add_argument("--input", default="labeled_dataset.json",
                        help="Labelled dataset")
    parser.add_argument("--output-dir", default="finetuned-specter2/",
                        help="Output directory for model + embeddings")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Pairs per batch (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--warmup-fraction", type=float, default=0.1,
                        help="Fraction of steps for linear warmup (default: 0.1)")
    parser.add_argument("--stride", type=int, default=256,
                        help="Chunk overlap (default: 256, matching generate_embeddings.py)")
    parser.add_argument("--max-chunks", type=int, default=8,
                        help="Max chunks per paper during training (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="InfoNCE temperature (default: 0.05)")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save checkpoint every N steps (default: 500)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set fraction (for split consistency)")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation set fraction (default: 0.1, excluded from training)")
    parser.add_argument("--print-sbatch", action="store_true",
                        help="Print SLURM sbatch template and exit")
    parser.add_argument("--hard-negatives", action="store_true",
                        help="Use category-aware batch sampling for harder in-batch negatives")
    parser.add_argument("--skip-regen", action="store_true",
                        help="Skip embedding regeneration after training")
    args = parser.parse_args()

    if args.print_sbatch:
        print(SBATCH_TEMPLATE)
        return

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device()
    print(f"Using device: {device}", file=sys.stderr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...", file=sys.stderr)
    records = load_dataset(Path(args.input))
    journals = [r["journal"] for r in records]
    print(f"Loaded {len(records)} records", file=sys.stderr)

    # Split (same as evaluation scripts)
    if args.val_size > 0:
        train_idx, _val_idx, test_idx = stratified_split_3way(
            journals, val_size=args.val_size, test_size=args.test_size,
            seed=args.seed)
        n_val = len(_val_idx)
        print(f"Train: {len(train_idx)}, Val: {n_val} (excluded), "
              f"Test: {len(test_idx)} (excluded)", file=sys.stderr)
    else:
        train_idx, test_idx = stratified_split(
            journals, test_size=args.test_size, seed=args.seed)
        print(f"Train: {len(train_idx)}, Test: {len(test_idx)} (excluded)",
              file=sys.stderr)

    # Build pair dataset
    pair_dataset = PairDataset(records, train_idx, seed=args.seed)
    print(f"Training pairs: {len(pair_dataset)}", file=sys.stderr)

    if args.hard_negatives:
        batch_sampler = HardNegativeBatchSampler(
            pair_dataset, records, args.batch_size, seed=args.seed)
        dataloader = DataLoader(
            pair_dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
        )
        print(f"Using hard negative batch sampling ({len(batch_sampler)} batches)",
              file=sys.stderr)
    else:
        dataloader = DataLoader(
            pair_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

    # Load model
    print("Loading SPECTER2 model + proximity adapter...", file=sys.stderr)
    from generate_embeddings import load_specter2
    tokenizer, model = load_specter2(device)

    # Freeze base model, train only adapter
    model.train_adapter("[PRX]")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.1f}%)", file=sys.stderr)

    # Optimiser and scheduler
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        remaining = total_steps - warmup_steps
        elapsed = step - warmup_steps
        return max(0.0, 1.0 - elapsed / max(1, remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    print(f"Total steps: {total_steps}, warmup: {warmup_steps}", file=sys.stderr)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if args.resume:
        ckpt = load_checkpoint(output_dir, model, optimiser, scheduler, device)
        if ckpt is not None:
            start_epoch = ckpt["epoch"]
            global_step = ckpt["step"]
            best_loss = ckpt["best_loss"]
        else:
            print("No checkpoint found, starting from scratch", file=sys.stderr)

    # Training loop
    avg_epoch_loss = float("nan")

    print(f"\nStarting training (epochs {start_epoch}-{args.epochs - 1})...",
          file=sys.stderr)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", file=sys.stderr)
        for batch_anchor_idx, batch_positive_idx in pbar:
            # Embed anchors and positives
            anchor_embs = []
            positive_embs = []

            for a_idx, p_idx in zip(batch_anchor_idx, batch_positive_idx):
                a_emb = embed_paper(
                    records[a_idx.item()], tokenizer, model, device,
                    stride=args.stride, max_chunks=args.max_chunks)
                p_emb = embed_paper(
                    records[p_idx.item()], tokenizer, model, device,
                    stride=args.stride, max_chunks=args.max_chunks)
                anchor_embs.append(a_emb)
                positive_embs.append(p_emb)

            anchor_embs = torch.stack(anchor_embs)
            positive_embs = torch.stack(positive_embs)

            # Compute loss
            loss = contrastive_loss(
                anchor_embs, positive_embs, temperature=args.temperature)

            # Backward + step
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimiser.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{epoch_loss / n_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            # Checkpoint
            if (args.checkpoint_every > 0
                    and global_step % args.checkpoint_every == 0):
                avg_loss = epoch_loss / n_batches
                save_checkpoint(
                    output_dir, model, optimiser, scheduler,
                    epoch, global_step, avg_loss, best_loss)

        avg_epoch_loss = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch}: avg_loss={avg_epoch_loss:.4f}", file=sys.stderr)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            # Save best adapter
            best_dir = output_dir / "best_adapter"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_adapter(str(best_dir), "[PRX]")
            print(f"  New best model saved (loss={best_loss:.4f})", file=sys.stderr)

        # End-of-epoch checkpoint
        save_checkpoint(
            output_dir, model, optimiser, scheduler,
            epoch + 1, global_step, avg_epoch_loss, best_loss)

    # Save training config
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_fraction": args.warmup_fraction,
        "temperature": args.temperature,
        "stride": args.stride,
        "max_chunks": args.max_chunks,
        "seed": args.seed,
        "n_pairs": len(pair_dataset),
        "total_steps": total_steps,
        "final_loss": avg_epoch_loss,
        "best_loss": best_loss,
        "n_trainable_params": n_trainable,
        "n_total_params": n_total,
        "hard_negatives": args.hard_negatives,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Regenerate embeddings
    if not args.skip_regen:
        # Load best adapter for regeneration
        best_dir = output_dir / "best_adapter"
        if best_dir.exists():
            print("Loading best adapter for embedding regeneration...",
                  file=sys.stderr)
            model.load_adapter(str(best_dir), set_active=True)

        regenerate_embeddings(
            records, tokenizer, model, device, output_dir,
            stride=args.stride)

        print(f"\nDone! Evaluate with:", file=sys.stderr)
        print(f"  python3 evaluate_knn.py "
              f"--embeddings-dir {output_dir}/embeddings", file=sys.stderr)
    else:
        print("\nSkipped embedding regeneration (--skip-regen).", file=sys.stderr)
        print(f"To regenerate later, load the adapter from "
              f"{output_dir}/best_adapter/", file=sys.stderr)


if __name__ == "__main__":
    main()
