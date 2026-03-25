#!/usr/bin/env python3
"""Train and save the journal prediction model for inference.

One-time script: instantiates JournalPredictor (trains from scratch),
saves all model artefacts to disk, then verifies the round-trip by
loading back and running a spot-check.

Usage:
  python3 save_model.py [--model-dir model]
"""

import argparse
import sys

import numpy as np

from predict_journal import JournalPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Train and save the journal prediction model")
    parser.add_argument("--model-dir", default="model",
                        help="Directory to save model artefacts (default: model)")
    parser.add_argument("--embeddings-dir",
                        default="finetuned-specter2-v2-hardneg/embeddings")
    parser.add_argument("--dataset", default="labeled_dataset.json")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--classifier-C", type=float, default=10.0)
    parser.add_argument("--min-papers", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=256,
                        help="PCA dimensions for classifier (default: 256)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Train from scratch
    print("Training model...", file=sys.stderr)
    predictor = JournalPredictor(
        embeddings_dir=args.embeddings_dir,
        dataset_path=args.dataset,
        alpha=args.alpha,
        classifier_C=args.classifier_C,
        pca_components=args.pca_components,
        min_papers=args.min_papers,
        seed=args.seed)

    # Save
    predictor.save(args.model_dir)

    # Verify round-trip: load back and spot-check
    print("\nVerifying round-trip...", file=sys.stderr)
    loaded = JournalPredictor.load(args.model_dir, args.dataset)

    # Check config values match
    assert loaded.alpha == predictor.alpha
    assert loaded.k == predictor.k
    assert loaded.T == predictor.T
    assert loaded.min_papers == predictor.min_papers
    assert len(loaded.eligible_journals) == len(predictor.eligible_journals)
    assert loaded.train_emb.shape == predictor.train_emb.shape
    assert np.allclose(loaded.train_emb[:5], predictor.train_emb[:5])

    # Score a few pool papers through predict_new and compare
    n_check = 5
    pool_emb = predictor.embeddings[predictor.pool_idx[:n_check]]
    pool_cats = [predictor.categories[i] for i in predictor.pool_idx[:n_check]]
    pool_dois = [predictor.dois[i] for i in predictor.pool_idx[:n_check]]

    new_results = loaded.predict_new(pool_emb, pool_cats, pool_dois)
    for i, result in enumerate(new_results):
        orig_preds, orig_info = predictor.predict(pool_index=i, top_k=10)
        orig_top = orig_preds[0][0]
        new_top = result["predictions"][0][0]
        orig_prob = orig_preds[0][1]
        new_prob = result["predictions"][0][1]
        match = "OK" if orig_top == new_top else "MISMATCH"
        print(f"  Paper {i}: top={new_top}  p={new_prob:.4f}  "
              f"(orig: {orig_top} p={orig_prob:.4f})  [{match}]",
              file=sys.stderr)

    print(f"\nModel saved to {args.model_dir}/ and verified.", file=sys.stderr)


if __name__ == "__main__":
    main()
