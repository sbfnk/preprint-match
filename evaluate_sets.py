#!/usr/bin/env python3
"""Evaluate conformal prediction sets on the test set.

For each paper, builds a prediction set by including journals in
descending probability order until the cumulative probability exceeds
a target coverage level (e.g., 90%). Then checks whether the true
journal falls within the set and reports set size statistics.

Usage:
  python3 evaluate_sets.py --embeddings-dir finetuned-specter2-hardneg/embeddings
  python3 evaluate_sets.py --coverage 0.80 0.90 0.95
"""

import argparse
import json
import sys
from collections import Counter

import numpy as np

from predict_journal import JournalPredictor


def compute_prediction_sets(proba, coverage_levels):
    """Compute prediction sets for each paper at given coverage levels.

    Args:
        proba: (n_papers, n_journals) calibrated probability matrix
        coverage_levels: list of target coverage levels (e.g., [0.80, 0.90, 0.95])

    Returns:
        dict mapping coverage level to array of set sizes (n_papers,)
    """
    n_papers, n_journals = proba.shape
    sorted_idx = np.argsort(proba, axis=1)[:, ::-1]  # descending
    sorted_probs = np.take_along_axis(proba, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)

    results = {}
    for level in coverage_levels:
        # For each paper, find how many journals needed to reach the level
        # searchsorted finds first index where cumsum >= level
        set_sizes = np.argmax(cumsum >= level, axis=1) + 1
        # If cumsum never reaches the level, include all journals
        never_reached = cumsum[:, -1] < level
        set_sizes[never_reached] = n_journals
        results[level] = set_sizes

    return results, sorted_idx


def evaluate_coverage(proba, true_indices, coverage_levels):
    """Evaluate prediction set coverage and set sizes on labelled data.

    Args:
        proba: (n_papers, n_journals) calibrated probability matrix
        true_indices: (n_papers,) true journal index for each paper
        coverage_levels: list of target coverage levels

    Returns:
        dict with results per coverage level
    """
    set_sizes_dict, sorted_idx = compute_prediction_sets(proba, coverage_levels)

    results = {}
    for level in coverage_levels:
        set_sizes = set_sizes_dict[level]

        # Check if true journal falls within the prediction set
        covered = np.zeros(len(true_indices), dtype=bool)
        for i, (true_idx, size) in enumerate(zip(true_indices, set_sizes)):
            prediction_set = set(sorted_idx[i, :size])
            covered[i] = true_idx in prediction_set

        empirical_coverage = float(covered.mean())
        results[level] = {
            "target_coverage": level,
            "empirical_coverage": empirical_coverage,
            "n_papers": len(true_indices),
            "n_covered": int(covered.sum()),
            "set_size_mean": float(set_sizes.mean()),
            "set_size_median": float(np.median(set_sizes)),
            "set_size_std": float(set_sizes.std()),
            "set_size_min": int(set_sizes.min()),
            "set_size_max": int(set_sizes.max()),
            "set_size_percentiles": {
                "p10": float(np.percentile(set_sizes, 10)),
                "p25": float(np.percentile(set_sizes, 25)),
                "p50": float(np.percentile(set_sizes, 50)),
                "p75": float(np.percentile(set_sizes, 75)),
                "p90": float(np.percentile(set_sizes, 90)),
                "p95": float(np.percentile(set_sizes, 95)),
            },
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate conformal prediction sets")
    parser.add_argument("--embeddings-dir",
                        default="finetuned-specter2-hardneg/embeddings")
    parser.add_argument("--dataset", default="labeled_dataset.json")
    parser.add_argument("--min-papers", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--coverage", type=float, nargs="+",
                        default=[0.50, 0.80, 0.90, 0.95, 0.99])
    parser.add_argument("--output", default="prediction_sets.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Training model and calibrating...", file=sys.stderr)
    predictor = JournalPredictor(
        embeddings_dir=args.embeddings_dir,
        dataset_path=args.dataset,
        alpha=args.alpha,
        min_papers=args.min_papers,
        seed=args.seed,
    )

    # Get test set probabilities and true labels
    n_val = len(predictor.val_idx)
    n_test = len(predictor.test_idx)
    # proba_pool = [val papers, test papers] — test starts at n_val
    proba_test = predictor.proba_pool[n_val:]

    # True journal indices in restricted class set (test papers only)
    test_true = []
    test_valid = []
    for i, j in enumerate(predictor.test_journals):
        idx = predictor.restricted_class_to_idx.get(j)
        if idx is not None:
            test_true.append(idx)
            test_valid.append(i)

    test_true = np.array(test_true)
    proba_test_valid = proba_test[test_valid]

    print(f"\nTest set: {len(test_valid)} papers with eligible journals "
          f"(out of {n_test} total)", file=sys.stderr)
    print(f"Coverage levels: {args.coverage}", file=sys.stderr)

    # Evaluate
    results = evaluate_coverage(proba_test_valid, test_true, args.coverage)

    # Per-tier breakdown (top-20, mid, tail journals by training count)
    journal_counts = predictor.journal_counts
    eligible = predictor.eligible_journals
    count_sorted = sorted(eligible, key=lambda j: -journal_counts[j])

    tiers = {
        "top_20": set(count_sorted[:20]),
        "top_50": set(count_sorted[:50]),
        "mid": set(count_sorted[50:150]),
        "tail": set(count_sorted[150:]),
    }

    for tier_name, tier_journals in tiers.items():
        tier_mask = np.array([
            predictor.restricted_classes[test_true[i]] in tier_journals
            for i in range(len(test_true))
        ])
        if tier_mask.sum() == 0:
            continue

        tier_results = evaluate_coverage(
            proba_test_valid[tier_mask], test_true[tier_mask], args.coverage)
        for level in args.coverage:
            results[level][f"tier_{tier_name}"] = {
                "n_papers": int(tier_mask.sum()),
                "empirical_coverage": tier_results[level]["empirical_coverage"],
                "set_size_median": tier_results[level]["set_size_median"],
                "set_size_mean": tier_results[level]["set_size_mean"],
            }

    # Print summary
    print(f"\n{'Coverage':>10} {'Empirical':>10} {'Median size':>12} "
          f"{'Mean size':>10} {'p90 size':>10}", file=sys.stderr)
    print("-" * 55, file=sys.stderr)
    for level in args.coverage:
        r = results[level]
        print(f"  {level:>6.0%}   {r['empirical_coverage']:>8.1%}   "
              f"{r['set_size_median']:>10.0f}   "
              f"{r['set_size_mean']:>8.1f}   "
              f"{r['set_size_percentiles']['p90']:>8.0f}", file=sys.stderr)

    print(f"\nPer-tier breakdown (90% coverage):", file=sys.stderr)
    r90 = results[0.90]
    for tier_name in ["top_20", "top_50", "mid", "tail"]:
        key = f"tier_{tier_name}"
        if key in r90:
            t = r90[key]
            print(f"  {tier_name:>8}: coverage={t['empirical_coverage']:.1%}  "
                  f"median_size={t['set_size_median']:.0f}  "
                  f"n={t['n_papers']}", file=sys.stderr)

    # Save
    output = {
        "config": {
            "min_papers": args.min_papers,
            "alpha": args.alpha,
            "n_eligible_journals": len(predictor.eligible_journals),
            "n_test_papers": len(test_valid),
            "coverage_levels": args.coverage,
        },
        "results": {str(k): v for k, v in results.items()},
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
