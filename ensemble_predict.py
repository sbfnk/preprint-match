#!/usr/bin/env python3
"""
Ensemble journal prediction: combine kNN + classifier via RRF or interpolation.

Runs both kNN (cosine similarity weighted voting) and logistic regression on
the same train/test split, then combines predictions using reciprocal rank
fusion (RRF) and/or score interpolation with optional alpha grid search.

Usage:
  python3 ensemble_predict.py --method both --output ensemble_results.json
  python3 ensemble_predict.py --method interpolation --alpha 0.4
  python3 ensemble_predict.py --method rrf --min-papers 10
"""

import json
import argparse
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from evaluate_knn import (
    load_embeddings,
    stratified_split,
    stratified_split_3way,
    cosine_similarity_chunked,
    predict_knn,
    evaluate,
    analyse_tiers,
    analyse_confusions,
    filter_by_min_papers,
)
from train_classifier import build_feature_matrix, proba_to_ranked_predictions


def softmax(scores):
    """Numerically stable softmax over a 1-D array of scores."""
    s = np.array(scores, dtype=np.float64)
    s -= s.max()
    exp_s = np.exp(s)
    return exp_s / exp_s.sum()


def reciprocal_rank_fusion(predictions_list, k=60):
    """Combine multiple ranked prediction lists via reciprocal rank fusion.

    For each test sample, computes RRF score per journal:
      sum over methods of 1 / (k + rank_in_method)
    Journals absent from a method contribute 0.

    Args:
        predictions_list: list of prediction lists (one per method).
            Each prediction list has one entry per test sample, where each
            entry is [(journal, score), ...] ranked by score.
        k: RRF constant (default 60).

    Returns:
        List of [(journal, rrf_score)] per test sample, sorted descending.
    """
    n_test = len(predictions_list[0])
    combined = []

    for i in range(n_test):
        rrf_scores = defaultdict(float)
        for preds in predictions_list:
            for rank, (journal, _score) in enumerate(preds[i]):
                rrf_scores[journal] += 1.0 / (k + rank + 1)
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        combined.append(ranked)

    return combined


def score_interpolation(knn_preds, clf_preds, alpha):
    """Combine kNN and classifier predictions via score interpolation.

    Normalises kNN similarity scores to probabilities via softmax, then
    combines: alpha * p_knn + (1 - alpha) * p_clf.

    Args:
        knn_preds: kNN predictions [(journal, cosine_sim_score), ...] per sample.
        clf_preds: classifier predictions [(journal, probability), ...] per sample.
        alpha: weight for kNN (0.0 = classifier only, 1.0 = kNN only).

    Returns:
        List of [(journal, combined_score)] per test sample, sorted descending.
    """
    n_test = len(knn_preds)
    combined = []

    for i in range(n_test):
        # Softmax-normalise kNN scores to probabilities
        knn_journals = [j for j, _ in knn_preds[i]]
        knn_scores = [s for _, s in knn_preds[i]]
        knn_probs = softmax(knn_scores)
        knn_dict = dict(zip(knn_journals, knn_probs))

        # Classifier probabilities (already normalised)
        clf_dict = dict(clf_preds[i])

        # Union of all journals
        all_journals = set(knn_dict) | set(clf_dict)
        merged = {}
        for j in all_journals:
            p_knn = knn_dict.get(j, 0.0)
            p_clf = clf_dict.get(j, 0.0)
            merged[j] = alpha * p_knn + (1.0 - alpha) * p_clf

        ranked = sorted(merged.items(), key=lambda x: -x[1])
        combined.append(ranked)

    return combined


def grid_search_alpha(knn_preds, clf_preds, true_journals, metric="mrr"):
    """Sweep alpha from 0.0 to 1.0 in 0.1 steps, return best alpha and grid.

    Returns:
        (best_alpha, grid_results) where grid_results is a dict mapping
        alpha -> {metric: value, ...}.
    """
    grid_results = {}
    best_alpha = 0.0
    best_score = -1.0

    for alpha_int in range(11):
        alpha = alpha_int / 10.0
        combined = score_interpolation(knn_preds, clf_preds, alpha)
        results = evaluate(combined, true_journals)
        grid_results[f"{alpha:.1f}"] = results

        score = results[metric]
        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha, grid_results


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble kNN + classifier for journal prediction")
    parser.add_argument("--embeddings-dir", default="embeddings/full-text-specter2",
                        help="Embeddings directory")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of kNN neighbours")
    parser.add_argument("--method", choices=["rrf", "interpolation", "both"],
                        default="both", help="Ensemble method (default: both)")
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="RRF constant (default: 60)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Interpolation weight for kNN (default: grid search 0.0-1.0)")
    parser.add_argument("--classifier-C", type=float, default=None,
                        help="Logistic regression C (default: grid search on val set)")
    parser.add_argument("--classifier-max-iter", type=int, default=200,
                        help="Max iterations for classifier (default: 200)")
    parser.add_argument("--no-category", action="store_true",
                        help="Disable category features in classifier")
    parser.add_argument("--min-papers", type=int, default=0,
                        help="Only evaluate journals with >= N training papers (default: 0 = all)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set fraction")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Validation set fraction for alpha tuning (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", default="ensemble_results.json",
                        help="Results output file")
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

    # Split
    use_val = args.val_size > 0
    if use_val:
        print("Splitting train/val/test...", file=sys.stderr)
        train_idx, val_idx, test_idx = stratified_split_3way(
            journals, val_size=args.val_size, test_size=args.test_size,
            seed=args.seed)
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, "
              f"Test: {len(test_idx)}", file=sys.stderr)
    else:
        print("Splitting train/test...", file=sys.stderr)
        train_idx, test_idx = stratified_split(
            journals, test_size=args.test_size, seed=args.seed)
        val_idx = np.array([], dtype=int)
        print(f"Train: {len(train_idx)}, Test: {len(test_idx)}", file=sys.stderr)

    train_emb = embeddings[train_idx]
    test_emb = embeddings[test_idx]
    train_journals = [journals[i] for i in train_idx]
    test_journals = [journals[i] for i in test_idx]
    train_categories = [categories[i] for i in train_idx]
    test_categories = [categories[i] for i in test_idx]

    if use_val:
        val_emb = embeddings[val_idx]
        val_journals = [journals[i] for i in val_idx]
        val_categories = [categories[i] for i in val_idx]

    n_train_journals = len(set(train_journals))
    n_test_journals = len(set(test_journals))
    print(f"Unique journals — train: {n_train_journals}, test: {n_test_journals}",
          file=sys.stderr)

    # --- kNN ---
    print(f"\nRunning kNN (k={args.k})...", file=sys.stderr)
    t0 = time.time()
    sim_matrix = cosine_similarity_chunked(test_emb, train_emb)
    knn_preds = predict_knn(sim_matrix, train_journals, k=args.k)
    knn_time = time.time() - t0
    print(f"kNN completed in {knn_time:.1f}s", file=sys.stderr)

    if use_val:
        val_sim_matrix = cosine_similarity_chunked(val_emb, train_emb)
        knn_preds_val = predict_knn(val_sim_matrix, train_journals, k=args.k)

    # --- Classifier ---
    # Build category encoder and feature matrices
    unique_cats = sorted(set(train_categories))
    cat_to_idx = {c: i + 1 for i, c in enumerate(unique_cats)}

    X_train = build_feature_matrix(
        train_emb, train_categories, cat_to_idx, use_category)
    X_test = build_feature_matrix(
        test_emb, test_categories, cat_to_idx, use_category)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_journals)

    if use_val:
        X_val = build_feature_matrix(
            val_emb, val_categories, cat_to_idx, use_category)

    # Grid search C on validation set if not specified
    if args.classifier_C is None and use_val:
        C_candidates = [0.01, 0.1, 1.0, 10.0]
        print(f"\nGrid searching classifier C on val set: {C_candidates}",
              file=sys.stderr)
        best_C = 1.0
        best_C_mrr = -1.0
        for C_val in C_candidates:
            clf_tmp = LogisticRegression(
                C=C_val, solver="lbfgs",
                max_iter=args.classifier_max_iter,
                random_state=args.seed,
            )
            clf_tmp.fit(X_train, y_train)
            proba_tmp = clf_tmp.predict_proba(X_val)
            preds_tmp = proba_to_ranked_predictions(
                proba_tmp, label_encoder.classes_)
            val_results = evaluate(preds_tmp, val_journals)
            mrr = val_results["mrr"]
            print(f"  C={C_val}: val MRR={mrr:.4f}, "
                  f"acc@1={val_results['accuracy@1']:.4f}, "
                  f"acc@10={val_results['accuracy@10']:.4f}", file=sys.stderr)
            if mrr > best_C_mrr:
                best_C_mrr = mrr
                best_C = C_val
        classifier_C = best_C
        print(f"  Best C={classifier_C} (val MRR={best_C_mrr:.4f})",
              file=sys.stderr)
    else:
        classifier_C = args.classifier_C if args.classifier_C is not None else 1.0

    print(f"\nTraining logistic regression (C={classifier_C})...",
          file=sys.stderr)
    clf = LogisticRegression(
        C=classifier_C,
        solver="lbfgs",
        max_iter=args.classifier_max_iter,
        random_state=args.seed,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    clf_time = time.time() - t0
    print(f"Classifier trained in {clf_time:.1f}s", file=sys.stderr)

    proba = clf.predict_proba(X_test)
    classes = label_encoder.classes_
    clf_preds = proba_to_ranked_predictions(proba, classes)

    if use_val:
        proba_val = clf.predict_proba(X_val)
        clf_preds_val = proba_to_ranked_predictions(proba_val, classes)

    # --- Optionally filter by min-papers ---
    if args.min_papers > 0:
        knn_preds, test_journals_knn, n_eligible = filter_by_min_papers(
            knn_preds, test_journals, train_journals, args.min_papers)
        clf_preds, test_journals_clf, _ = filter_by_min_papers(
            clf_preds, test_journals, train_journals, args.min_papers)
        assert test_journals_knn == test_journals_clf, (
            "kNN and classifier filtered test sets are misaligned")
        test_journals = test_journals_knn
        print(f"\n--min-papers={args.min_papers}: {n_eligible} eligible journals, "
              f"{len(test_journals)} test papers retained "
              f"(excluded {len(test_idx) - len(test_journals)})", file=sys.stderr)

        if use_val:
            knn_preds_val, val_journals_knn, _ = filter_by_min_papers(
                knn_preds_val, val_journals, train_journals, args.min_papers)
            clf_preds_val, val_journals_clf, _ = filter_by_min_papers(
                clf_preds_val, val_journals, train_journals, args.min_papers)
            assert val_journals_knn == val_journals_clf
            val_journals = val_journals_knn

    # --- Evaluate individual methods ---
    print("\n=== kNN-only results ===", file=sys.stderr)
    knn_results = evaluate(knn_preds, test_journals)
    for metric, value in knn_results.items():
        if metric != "n_test":
            print(f"  {metric}: {value:.4f}", file=sys.stderr)

    print("\n=== Classifier-only results ===", file=sys.stderr)
    clf_results = evaluate(clf_preds, test_journals)
    for metric, value in clf_results.items():
        if metric != "n_test":
            print(f"  {metric}: {value:.4f}", file=sys.stderr)

    # --- Ensemble: RRF ---
    rrf_results = None
    if args.method in ("rrf", "both"):
        print(f"\n=== RRF (k={args.rrf_k}) ===", file=sys.stderr)
        rrf_preds = reciprocal_rank_fusion(
            [knn_preds, clf_preds], k=args.rrf_k)
        rrf_results = evaluate(rrf_preds, test_journals)
        for metric, value in rrf_results.items():
            if metric != "n_test":
                print(f"  {metric}: {value:.4f}", file=sys.stderr)

    # --- Ensemble: Interpolation ---
    interp_results = None
    interp_grid = None
    best_alpha = None
    if args.method in ("interpolation", "both"):
        if args.alpha is not None:
            print(f"\n=== Interpolation (alpha={args.alpha:.1f}) ===",
                  file=sys.stderr)
            interp_preds = score_interpolation(knn_preds, clf_preds, args.alpha)
            interp_results = evaluate(interp_preds, test_journals)
            best_alpha = args.alpha
            for metric, value in interp_results.items():
                if metric != "n_test":
                    print(f"  {metric}: {value:.4f}", file=sys.stderr)
        else:
            if use_val:
                # Tune alpha on validation set, evaluate on test set
                print("\n=== Interpolation (grid search on val set) ===",
                      file=sys.stderr)
                best_alpha, interp_grid = grid_search_alpha(
                    knn_preds_val, clf_preds_val, val_journals)
                print(f"  Best alpha: {best_alpha:.1f} (tuned on val set)",
                      file=sys.stderr)

                print("\n  Alpha grid (val set):", file=sys.stderr)
                for alpha_str, res in sorted(interp_grid.items()):
                    print(f"    alpha={alpha_str}: "
                          f"acc@1={res['accuracy@1']:.4f}  "
                          f"acc@10={res['accuracy@10']:.4f}  "
                          f"mrr={res['mrr']:.4f}", file=sys.stderr)

                # Final evaluation on test set with best alpha
                print(f"\n=== Test set results (alpha={best_alpha:.1f}) ===",
                      file=sys.stderr)
                interp_preds = score_interpolation(
                    knn_preds, clf_preds, best_alpha)
                interp_results = evaluate(interp_preds, test_journals)
                for metric, value in interp_results.items():
                    if metric != "n_test":
                        print(f"  {metric}: {value:.4f}", file=sys.stderr)
            else:
                print("\n=== Interpolation (grid search) ===", file=sys.stderr)
                best_alpha, interp_grid = grid_search_alpha(
                    knn_preds, clf_preds, test_journals)
                interp_results = interp_grid[f"{best_alpha:.1f}"]
                print(f"  Best alpha: {best_alpha:.1f} "
                      f"(tuned on test set — optimistic)", file=sys.stderr)
                for metric, value in interp_results.items():
                    if metric != "n_test":
                        print(f"  {metric}: {value:.4f}", file=sys.stderr)

                print("\n  Alpha grid:", file=sys.stderr)
                for alpha_str, res in sorted(interp_grid.items()):
                    print(f"    alpha={alpha_str}: "
                          f"acc@1={res['accuracy@1']:.4f}  "
                          f"acc@10={res['accuracy@10']:.4f}  "
                          f"mrr={res['mrr']:.4f}", file=sys.stderr)

    # --- Per-tier breakdown for best ensemble ---
    best_ensemble_preds = None
    best_method_name = None
    if rrf_results and interp_results:
        if rrf_results["mrr"] >= interp_results["mrr"]:
            best_ensemble_preds = rrf_preds
            best_method_name = "rrf"
        else:
            best_ensemble_preds = score_interpolation(
                knn_preds, clf_preds, best_alpha)
            best_method_name = f"interpolation (alpha={best_alpha:.1f})"
    elif rrf_results:
        best_ensemble_preds = rrf_preds
        best_method_name = "rrf"
    elif interp_results:
        best_ensemble_preds = score_interpolation(
            knn_preds, clf_preds, best_alpha)
        best_method_name = f"interpolation (alpha={best_alpha:.1f})"

    if best_ensemble_preds:
        print(f"\n=== Per-tier breakdown (best ensemble: {best_method_name}) ===",
              file=sys.stderr)
        tier_results = analyse_tiers(
            best_ensemble_preds, test_journals, train_journals)
        for tier in ["top-20", "top-50", "mid-tail", "long-tail"]:
            if tier in tier_results:
                r = tier_results[tier]
                print(f"  {tier} (n={r['n_test']}):", file=sys.stderr)
                for metric, value in r.items():
                    if metric != "n_test":
                        print(f"    {metric}: {value:.4f}", file=sys.stderr)

    # --- Save results ---
    results = {
        "config": {
            "embeddings_dir": str(emb_dir),
            "k": args.k,
            "method": args.method,
            "rrf_k": args.rrf_k,
            "classifier_C": classifier_C,
            "classifier_max_iter": args.classifier_max_iter,
            "use_category": use_category,
            "min_papers": args.min_papers,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "seed": args.seed,
            "n_train": len(train_idx),
            "n_val": len(val_idx) if use_val else 0,
            "n_test": len(test_idx),
            "n_test_after_filter": len(test_journals),
            "n_train_journals": n_train_journals,
            "n_test_journals": n_test_journals,
            "knn_time_s": round(knn_time, 1),
            "clf_time_s": round(clf_time, 1),
        },
        "knn_only": knn_results,
        "classifier_only": clf_results,
    }

    if rrf_results:
        results["rrf"] = rrf_results
    if interp_results:
        results["interpolation"] = {
            "best_alpha": best_alpha,
            "results": interp_results,
        }
        if interp_grid:
            results["interpolation"]["grid"] = interp_grid
            if use_val:
                results["interpolation"]["tuned_on"] = "validation set"
            else:
                results["interpolation"]["caveat"] = (
                    "Alpha tuned on test set — results are optimistic. "
                    "Use --val-size for unbiased estimates."
                )

    if best_ensemble_preds:
        results["best_ensemble"] = {
            "method": best_method_name,
            "per_tier": tier_results,
        }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
