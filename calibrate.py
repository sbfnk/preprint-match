#!/usr/bin/env python3
"""Calibration analysis and temperature scaling for ensemble journal predictions.

Assesses calibration quality via reliability diagrams and Expected Calibration
Error (ECE), then fits a single temperature parameter T on the validation set
to produce well-calibrated probabilities.

The fitted temperature can be used by downstream tools (journal-as-a-filter,
recommendation) to convert raw ensemble scores into meaningful probabilities.

Usage:
  python3 calibrate.py
  python3 calibrate.py --embeddings-dir embeddings/full-text-specter2
  python3 calibrate.py --output calibration_original.json
"""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from evaluate_knn import (
    load_embeddings,
    stratified_split_3way,
    cosine_similarity_chunked,
    predict_knn,
)
from train_classifier import build_feature_matrix
from ensemble_predict import softmax


def ensemble_proba_matrix(knn_preds, clf_proba, classes, alpha):
    """Build full probability matrix from kNN predictions and classifier probabilities.

    Args:
        knn_preds: list of [(journal, score), ...] per sample (k entries each).
        clf_proba: array (n_samples, n_classes) of classifier probabilities.
        classes: array of class labels corresponding to clf_proba columns.
        alpha: kNN weight (0 = classifier only, 1 = kNN only).

    Returns:
        array (n_samples, n_classes) of ensemble probabilities.
    """
    n_samples, n_classes = clf_proba.shape
    class_to_idx = {c: i for i, c in enumerate(classes)}

    proba = (1 - alpha) * clf_proba.copy()

    for i in range(n_samples):
        knn_journals = [j for j, _ in knn_preds[i]]
        knn_scores = np.array([s for _, s in knn_preds[i]])
        knn_probs = softmax(knn_scores)

        for j_name, p in zip(knn_journals, knn_probs):
            idx = class_to_idx.get(j_name)
            if idx is not None:
                proba[i, idx] += alpha * p

    return proba


def reliability_diagram(proba, true_indices, n_bins=15):
    """Compute top-1 reliability diagram.

    Bins test samples by confidence (max predicted probability), then checks
    whether actual accuracy in each bin matches the mean confidence.

    Returns:
        list of dicts with bin_lower, bin_upper, count, mean_confidence,
        mean_accuracy, gap.
    """
    confidences = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    correct = (predictions == true_indices).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())

        if count > 0:
            mean_conf = float(confidences[mask].mean())
            mean_acc = float(correct[mask].mean())
        else:
            mean_conf = float((lo + hi) / 2)
            mean_acc = 0.0

        bins.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "count": count,
            "mean_confidence": mean_conf,
            "mean_accuracy": mean_acc,
            "gap": abs(mean_acc - mean_conf),
        })

    return bins


def compute_ece(bins):
    """Expected Calibration Error: weighted average of |confidence - accuracy|."""
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    return sum(b["count"] * b["gap"] for b in bins) / total


def compute_mce(bins):
    """Maximum Calibration Error: worst-case bin gap."""
    non_empty = [b for b in bins if b["count"] > 0]
    if not non_empty:
        return 0.0
    return max(b["gap"] for b in non_empty)


def temperature_scale(proba, T):
    """Apply temperature scaling: log, divide by T, re-softmax."""
    log_p = np.log(np.maximum(proba, 1e-30))
    scaled = log_p / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def fit_temperature(proba, true_indices, bounds=(0.01, 20.0)):
    """Find optimal temperature T minimising NLL on calibration data.

    Returns:
        (T_optimal, nll_at_optimum)
    """
    def nll(T):
        calibrated = temperature_scale(proba, T)
        true_probs = calibrated[np.arange(len(true_indices)), true_indices]
        return -np.mean(np.log(np.maximum(true_probs, 1e-30)))

    result = minimize_scalar(nll, bounds=bounds, method='bounded')
    return result.x, result.fun


def compute_nll(proba, true_indices):
    """Negative log-likelihood of the true class."""
    true_probs = proba[np.arange(len(true_indices)), true_indices]
    return -np.mean(np.log(np.maximum(true_probs, 1e-30)))


def confidence_stats(proba):
    """Summary statistics for top-1 confidence distribution."""
    confs = np.max(proba, axis=1)
    return {
        "mean": float(confs.mean()),
        "median": float(np.median(confs)),
        "std": float(confs.std()),
        "min": float(confs.min()),
        "max": float(confs.max()),
        "pct_above_10": float((confs > 0.1).mean()),
        "pct_above_20": float((confs > 0.2).mean()),
        "pct_above_50": float((confs > 0.5).mean()),
    }


def print_reliability(bins, label, file=sys.stderr):
    """Print reliability diagram to stderr."""
    print(f"\nReliability diagram ({label}):", file=file)
    for b in bins:
        if b["count"] > 0:
            print(f"  [{b['bin_lower']:.3f}, {b['bin_upper']:.3f}]: "
                  f"n={b['count']:5d}  conf={b['mean_confidence']:.4f}  "
                  f"acc={b['mean_accuracy']:.4f}  gap={b['gap']:.4f}",
                  file=file)


def main():
    parser = argparse.ArgumentParser(
        description="Calibration analysis for ensemble predictions")
    parser.add_argument("--embeddings-dir", default="finetuned-specter2/embeddings",
                        help="Embeddings directory (default: fine-tuned)")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of kNN neighbours")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Interpolation weight for kNN (default: 0.3)")
    parser.add_argument("--classifier-C", type=float, default=1.0,
                        help="Logistic regression regularisation")
    parser.add_argument("--classifier-max-iter", type=int, default=200,
                        help="Max iterations for classifier")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation set fraction")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n-bins", type=int, default=15,
                        help="Number of reliability diagram bins")
    parser.add_argument("--output", default="calibration_results.json",
                        help="Results output file")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)

    # Load data
    print("Loading embeddings...", file=sys.stderr)
    embeddings, metadata = load_embeddings(emb_dir)
    journals = metadata["journals"]
    categories = metadata["categories"]
    print(f"Loaded {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim)",
          file=sys.stderr)

    # 3-way split
    train_idx, val_idx, test_idx = stratified_split_3way(
        journals, val_size=args.val_size, test_size=args.test_size,
        seed=args.seed)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}",
          file=sys.stderr)

    train_emb = embeddings[train_idx]
    val_emb = embeddings[val_idx]
    test_emb = embeddings[test_idx]
    train_journals = [journals[i] for i in train_idx]
    val_journals = [journals[i] for i in val_idx]
    test_journals = [journals[i] for i in test_idx]
    train_categories = [categories[i] for i in train_idx]
    val_categories = [categories[i] for i in val_idx]
    test_categories = [categories[i] for i in test_idx]

    # --- kNN ---
    print("Running kNN...", file=sys.stderr)
    val_sim = cosine_similarity_chunked(val_emb, train_emb)
    knn_preds_val = predict_knn(val_sim, train_journals, k=args.k)
    test_sim = cosine_similarity_chunked(test_emb, train_emb)
    knn_preds_test = predict_knn(test_sim, train_journals, k=args.k)

    # --- Classifier ---
    print("Training classifier...", file=sys.stderr)
    unique_cats = sorted(set(train_categories))
    cat_to_idx = {c: i + 1 for i, c in enumerate(unique_cats)}

    X_train = build_feature_matrix(train_emb, train_categories, cat_to_idx, True)
    X_val = build_feature_matrix(val_emb, val_categories, cat_to_idx, True)
    X_test = build_feature_matrix(test_emb, test_categories, cat_to_idx, True)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_journals)
    classes = label_encoder.classes_

    clf = LogisticRegression(
        C=args.classifier_C, solver="lbfgs",
        max_iter=args.classifier_max_iter, random_state=args.seed)
    clf.fit(X_train, y_train)
    print("Classifier trained.", file=sys.stderr)

    clf_proba_val = clf.predict_proba(X_val)
    clf_proba_test = clf.predict_proba(X_test)

    # --- Build ensemble probability matrices ---
    print("Building ensemble probability matrices...", file=sys.stderr)
    proba_val = ensemble_proba_matrix(
        knn_preds_val, clf_proba_val, classes, args.alpha)
    proba_test = ensemble_proba_matrix(
        knn_preds_test, clf_proba_test, classes, args.alpha)

    # True labels as class indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    val_true = np.array([class_to_idx[j] for j in val_journals])
    test_true = np.array([class_to_idx[j] for j in test_journals])

    # === Uncalibrated analysis ===
    print("\n=== Uncalibrated (val set) ===", file=sys.stderr)
    bins_val_uncal = reliability_diagram(proba_val, val_true, args.n_bins)
    ece_val_uncal = compute_ece(bins_val_uncal)
    mce_val_uncal = compute_mce(bins_val_uncal)
    nll_val_uncal = compute_nll(proba_val, val_true)
    top1_acc_val = float((np.argmax(proba_val, axis=1) == val_true).mean())
    print(f"Top-1 accuracy: {top1_acc_val:.4f}", file=sys.stderr)
    print(f"ECE: {ece_val_uncal:.4f}", file=sys.stderr)
    print(f"MCE: {mce_val_uncal:.4f}", file=sys.stderr)
    print(f"NLL: {nll_val_uncal:.4f}", file=sys.stderr)
    print_reliability(bins_val_uncal, "uncalibrated, val")

    print("\n=== Uncalibrated (test set) ===", file=sys.stderr)
    bins_test_uncal = reliability_diagram(proba_test, test_true, args.n_bins)
    ece_test_uncal = compute_ece(bins_test_uncal)
    mce_test_uncal = compute_mce(bins_test_uncal)
    nll_test_uncal = compute_nll(proba_test, test_true)
    top1_acc_test = float((np.argmax(proba_test, axis=1) == test_true).mean())
    print(f"Top-1 accuracy: {top1_acc_test:.4f}", file=sys.stderr)
    print(f"ECE: {ece_test_uncal:.4f}", file=sys.stderr)
    print(f"MCE: {mce_test_uncal:.4f}", file=sys.stderr)
    print(f"NLL: {nll_test_uncal:.4f}", file=sys.stderr)
    print_reliability(bins_test_uncal, "uncalibrated, test")

    # === Fit temperature on val set ===
    print("\n=== Temperature scaling ===", file=sys.stderr)
    T_opt, nll_val_cal = fit_temperature(proba_val, val_true)
    print(f"Optimal T: {T_opt:.4f}", file=sys.stderr)
    print(f"Val NLL: {nll_val_uncal:.4f} -> {nll_val_cal:.4f}", file=sys.stderr)

    # Apply to both sets
    proba_val_cal = temperature_scale(proba_val, T_opt)
    proba_test_cal = temperature_scale(proba_test, T_opt)

    # === Calibrated analysis (val) ===
    print("\n=== Calibrated (val set, T={:.4f}) ===".format(T_opt), file=sys.stderr)
    bins_val_cal = reliability_diagram(proba_val_cal, val_true, args.n_bins)
    ece_val_cal = compute_ece(bins_val_cal)
    mce_val_cal = compute_mce(bins_val_cal)
    print(f"ECE: {ece_val_cal:.4f} (was {ece_val_uncal:.4f})", file=sys.stderr)
    print(f"MCE: {mce_val_cal:.4f} (was {mce_val_uncal:.4f})", file=sys.stderr)
    print_reliability(bins_val_cal, "calibrated, val")

    # === Calibrated analysis (test) ===
    print("\n=== Calibrated (test set, T={:.4f}) ===".format(T_opt), file=sys.stderr)
    bins_test_cal = reliability_diagram(proba_test_cal, test_true, args.n_bins)
    ece_test_cal = compute_ece(bins_test_cal)
    mce_test_cal = compute_mce(bins_test_cal)
    nll_test_cal = compute_nll(proba_test_cal, test_true)
    print(f"ECE: {ece_test_cal:.4f} (was {ece_test_uncal:.4f})", file=sys.stderr)
    print(f"MCE: {mce_test_cal:.4f} (was {mce_test_uncal:.4f})", file=sys.stderr)
    print(f"NLL: {nll_test_cal:.4f} (was {nll_test_uncal:.4f})", file=sys.stderr)
    print_reliability(bins_test_cal, "calibrated, test")

    # === Save results ===
    results = {
        "config": {
            "embeddings_dir": str(emb_dir),
            "alpha": args.alpha,
            "k": args.k,
            "n_bins": args.n_bins,
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "n_classes": int(len(classes)),
        },
        "temperature": T_opt,
        "validation": {
            "uncalibrated": {
                "top1_accuracy": top1_acc_val,
                "ece": ece_val_uncal,
                "mce": mce_val_uncal,
                "nll": nll_val_uncal,
                "reliability_diagram": bins_val_uncal,
                "confidence_distribution": confidence_stats(proba_val),
            },
            "calibrated": {
                "top1_accuracy": top1_acc_val,
                "ece": ece_val_cal,
                "mce": mce_val_cal,
                "nll": nll_val_cal,
                "reliability_diagram": bins_val_cal,
                "confidence_distribution": confidence_stats(proba_val_cal),
            },
        },
        "test": {
            "uncalibrated": {
                "top1_accuracy": top1_acc_test,
                "ece": ece_test_uncal,
                "mce": mce_test_uncal,
                "nll": nll_test_uncal,
                "reliability_diagram": bins_test_uncal,
                "confidence_distribution": confidence_stats(proba_test),
            },
            "calibrated": {
                "top1_accuracy": top1_acc_test,
                "ece": ece_test_cal,
                "mce": mce_test_cal,
                "nll": nll_test_cal,
                "reliability_diagram": bins_test_cal,
                "confidence_distribution": confidence_stats(proba_test_cal),
            },
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}", file=sys.stderr)
    print(f"\nTemperature T={T_opt:.4f} — use this value in downstream tools.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
