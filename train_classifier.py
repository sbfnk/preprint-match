#!/usr/bin/env python3
"""
Train a classifier for journal prediction using pre-computed embeddings.

Trains logistic regression or MLP on SPECTER2 embeddings + medRxiv
category one-hot features. Uses the same stratified split and metrics as
evaluate_knn.py.

Usage:
  python3 train_classifier.py --output classifier_results_logistic.json
  python3 train_classifier.py --no-category --output classifier_results_logistic_no_cat.json
  python3 train_classifier.py --model mlp --output classifier_results_mlp.json
"""

import json
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from evaluate_knn import (
    load_embeddings,
    stratified_split,
    evaluate,
    analyse_tiers,
    analyse_confusions,
)


def build_feature_matrix(embeddings, categories, cat_to_idx, use_category=True):
    """Build feature matrix: L2-normalised embeddings + optional category one-hot.

    Returns a dense array (LogisticRegression with lbfgs requires it).
    """
    # L2-normalise embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    emb_normed = embeddings / norms

    if not use_category:
        return emb_normed

    # One-hot encode categories (index 0 reserved for unknown)
    n = len(categories)
    n_cats = max(cat_to_idx.values()) + 1
    cat_onehot = np.zeros((n, n_cats), dtype=np.float32)
    for i, c in enumerate(categories):
        cat_onehot[i, cat_to_idx.get(c, 0)] = 1.0

    return np.hstack([emb_normed, cat_onehot])


def proba_to_ranked_predictions(proba, classes, top_k=50):
    """Convert probability matrix to ranked [(journal, score)] predictions.

    Uses argpartition for efficient top-k selection.
    """
    n = proba.shape[0]
    predictions = []

    k = min(top_k, proba.shape[1])

    for i in range(n):
        row = proba[i]
        top_idx = np.argpartition(row, -k)[-k:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        ranked = [(classes[j], float(row[j])) for j in top_idx]
        predictions.append(ranked)

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Train classifier for journal prediction")
    parser.add_argument("--embeddings-dir", default="embeddings/full-text-specter2",
                        help="Embeddings directory")
    parser.add_argument("--model", choices=["logistic", "mlp"], default="logistic",
                        help="Classifier type (default: logistic)")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Regularisation strength (default: 1.0)")
    parser.add_argument("--no-category", action="store_true",
                        help="Disable category features")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", default="classifier_results.json",
                        help="Results output file")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Maximum iterations for solver (default: 200)")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    use_category = not args.no_category

    # Load data
    print("Loading embeddings...", file=sys.stderr)
    embeddings, metadata = load_embeddings(emb_dir)
    journals = metadata["journals"]
    categories = metadata["categories"]
    print(f"Loaded {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim)",
          file=sys.stderr)

    # Split (identical to kNN)
    print("Splitting train/test...", file=sys.stderr)
    train_idx, test_idx = stratified_split(
        journals, test_size=args.test_size, seed=args.seed)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}", file=sys.stderr)

    train_emb = embeddings[train_idx]
    test_emb = embeddings[test_idx]
    train_journals = [journals[i] for i in train_idx]
    test_journals = [journals[i] for i in test_idx]
    train_categories = [categories[i] for i in train_idx]
    test_categories = [categories[i] for i in test_idx]

    n_train_journals = len(set(train_journals))
    n_test_journals = len(set(test_journals))
    print(f"Unique journals — train: {n_train_journals}, test: {n_test_journals}",
          file=sys.stderr)

    # Build category encoder from training data
    unique_cats = sorted(set(train_categories))
    cat_to_idx = {c: i + 1 for i, c in enumerate(unique_cats)}  # 0 = unknown
    print(f"Categories: {len(unique_cats)}", file=sys.stderr)
    print(f"Features: embeddings ({train_emb.shape[1]})"
          + (f" + category ({len(unique_cats)})" if use_category else ""),
          file=sys.stderr)

    # Encode journal labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_journals)

    # Build feature matrices
    print("Building feature matrices...", file=sys.stderr)
    X_train = build_feature_matrix(
        train_emb, train_categories, cat_to_idx, use_category)
    X_test = build_feature_matrix(
        test_emb, test_categories, cat_to_idx, use_category)
    print(f"Feature matrix shape: {X_train.shape}", file=sys.stderr)

    # Train classifier
    if args.model == "logistic":
        print(f"Training logistic regression (C={args.C}, "
              f"max_iter={args.max_iter})...", file=sys.stderr)
        clf = LogisticRegression(
            C=args.C,
            solver="lbfgs",
            max_iter=args.max_iter,
            random_state=args.seed,
        )
    else:
        print(f"Training MLP (C={args.C}, max_iter={args.max_iter})...",
              file=sys.stderr)
        clf = MLPClassifier(
            hidden_layer_sizes=(256,),
            alpha=1.0 / args.C,
            max_iter=args.max_iter,
            random_state=args.seed,
            early_stopping=True,
            validation_fraction=0.1,
        )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training took {train_time:.1f}s", file=sys.stderr)

    # Predict probabilities
    print("Predicting...", file=sys.stderr)
    proba = clf.predict_proba(X_test)
    classes = label_encoder.classes_

    # Convert to ranked predictions
    predictions = proba_to_ranked_predictions(proba, classes)

    # Evaluate
    print("\nOverall results:", file=sys.stderr)
    overall = evaluate(predictions, test_journals)
    for metric, value in overall.items():
        if metric != "n_test":
            print(f"  {metric}: {value:.4f}", file=sys.stderr)
    print(f"  n_test: {overall['n_test']}", file=sys.stderr)

    # Per-tier results
    print("\nPer-tier results:", file=sys.stderr)
    tier_results = analyse_tiers(predictions, test_journals, train_journals)
    for tier in ["top-20", "top-50", "mid-tail", "long-tail"]:
        if tier in tier_results:
            r = tier_results[tier]
            print(f"  {tier} (n={r['n_test']}):", file=sys.stderr)
            for metric, value in r.items():
                if metric != "n_test":
                    print(f"    {metric}: {value:.4f}", file=sys.stderr)

    # Confusion analysis
    print("\nTop confusion pairs (true -> predicted):", file=sys.stderr)
    confusions = analyse_confusions(predictions, test_journals)
    for (true_j, pred_j), count in confusions:
        print(f"  {count:3d}x  {true_j}  ->  {pred_j}", file=sys.stderr)

    # Save results
    results = {
        "config": {
            "model": args.model,
            "features": ("embeddings+category" if use_category
                         else "embeddings"),
            "C": args.C,
            "max_iter": args.max_iter,
            "test_size": args.test_size,
            "seed": args.seed,
            "embeddings_dir": str(emb_dir),
            "n_features": X_train.shape[1],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_journals": n_train_journals,
            "n_test_journals": n_test_journals,
            "n_categories": len(unique_cats),
            "train_time_s": round(train_time, 1),
        },
        "overall": overall,
        "per_tier": tier_results,
        "top_confusions": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in confusions
        ],
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
