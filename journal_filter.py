#!/usr/bin/env python3
"""Journal-as-a-filter: find preprints predicted for a specific journal.

Given a target journal, scans a pool of preprints and returns those the
ensemble model predicts are likely to be published in that journal,
ranked by predicted probability.

Usage:
  python3 journal_filter.py "The Lancet Infectious Diseases"
  python3 journal_filter.py "PLOS ONE" --top-k 50 --threshold 0.05
  python3 journal_filter.py --list-journals
  python3 journal_filter.py --interactive
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from evaluate_knn import (
    load_embeddings,
    stratified_split_3way,
    cosine_similarity_chunked,
    predict_knn,
)
from train_classifier import build_feature_matrix
from calibrate import ensemble_proba_matrix, temperature_scale


def load_titles(dataset_path, dois):
    """Load titles from labeled_dataset.json, indexed by DOI."""
    doi_set = set(dois)
    titles = {}

    print("Loading paper titles...", file=sys.stderr)
    with open(dataset_path) as f:
        dataset = json.load(f)

    for paper in dataset:
        doi = paper.get("preprint_doi", "")
        if doi in doi_set:
            titles[doi] = paper.get("title", "(no title)")

    return titles


def resolve_journal(query, journal_counts, min_papers=0):
    """Resolve a journal name query (exact, then case-insensitive substring).

    Returns (resolved_name, n_training_papers) or (None, 0).
    """
    # Exact match
    if query in journal_counts:
        count = journal_counts[query]
        if count >= min_papers:
            return query, count

    # Case-insensitive exact
    query_lower = query.lower()
    for name, count in journal_counts.items():
        if name.lower() == query_lower and count >= min_papers:
            return name, count

    # Substring match
    matches = []
    for name, count in journal_counts.items():
        if query_lower in name.lower() and count >= min_papers:
            matches.append((name, count))

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"\nAmbiguous query '{query}'. Did you mean:", file=sys.stderr)
        for name, count in sorted(matches, key=lambda x: -x[1])[:15]:
            print(f"  {count:4d} papers  {name}", file=sys.stderr)
        return None, 0

    return None, 0


def format_results(papers, top_k=20, threshold=0.0):
    """Format filtered papers for display."""
    filtered = [(doi, title, prob) for doi, title, prob in papers
                if prob >= threshold]
    filtered = filtered[:top_k]
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Find preprints predicted for a specific journal")
    parser.add_argument("journal", nargs="?", default=None,
                        help="Target journal name (or substring)")
    parser.add_argument("--embeddings-dir", default="finetuned-specter2/embeddings",
                        help="Embeddings directory")
    parser.add_argument("--dataset", default="labeled_dataset.json",
                        help="Dataset file for paper titles")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Show top-k predictions (default: 20)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum probability threshold (default: 0, show all)")
    parser.add_argument("--min-papers", type=int, default=5,
                        help="Minimum training papers for a journal to be supported (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Ensemble interpolation weight (default: 0.3)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for calibration (default: load from calibration_results.json)")
    parser.add_argument("--k", type=int, default=20,
                        help="kNN neighbours (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-journals", action="store_true",
                        help="List supported journals and exit")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: query multiple journals")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)

    # Load temperature from calibration results if available
    T = args.temperature
    if T is None:
        cal_path = Path("calibration_results.json")
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            T = cal.get("temperature", 1.0)
            print(f"Loaded temperature T={T:.4f} from {cal_path}",
                  file=sys.stderr)
        else:
            T = 1.0

    # Load embeddings
    print("Loading embeddings...", file=sys.stderr)
    embeddings, metadata = load_embeddings(emb_dir)
    journals = metadata["journals"]
    categories = metadata["categories"]
    dois = metadata["dois"]

    # Split: train on training set, predict on val+test (the "pool")
    train_idx, val_idx, test_idx = stratified_split_3way(
        journals, val_size=0.1, test_size=0.2, seed=args.seed)
    pool_idx = np.concatenate([val_idx, test_idx])

    train_journals = [journals[i] for i in train_idx]
    pool_journals = [journals[i] for i in pool_idx]
    pool_dois = [dois[i] for i in pool_idx]
    pool_categories = [categories[i] for i in pool_idx]
    train_categories = [categories[i] for i in train_idx]

    journal_counts = Counter(train_journals)
    n_supported = sum(1 for c in journal_counts.values() if c >= args.min_papers)

    print(f"Training set: {len(train_idx)} papers, {len(journal_counts)} journals",
          file=sys.stderr)
    print(f"Prediction pool: {len(pool_idx)} papers (val + test)",
          file=sys.stderr)
    print(f"Supported journals (>={args.min_papers} papers): {n_supported}",
          file=sys.stderr)

    # List journals mode
    if args.list_journals:
        print(f"\nSupported journals (>={args.min_papers} training papers):\n")
        supported = [(name, count) for name, count in journal_counts.items()
                     if count >= args.min_papers]
        for name, count in sorted(supported, key=lambda x: -x[1]):
            print(f"  {count:4d}  {name}")
        print(f"\nTotal: {len(supported)} journals")
        return

    # Need to compute predictions — train model
    print("\nTraining model...", file=sys.stderr)
    train_emb = embeddings[train_idx]
    pool_emb = embeddings[pool_idx]

    # kNN
    print("  Computing kNN similarities...", file=sys.stderr)
    pool_sim = cosine_similarity_chunked(pool_emb, train_emb)
    knn_preds = predict_knn(pool_sim, train_journals, k=args.k)

    # Classifier
    print("  Training classifier...", file=sys.stderr)
    unique_cats = sorted(set(train_categories))
    cat_to_idx = {c: i + 1 for i, c in enumerate(unique_cats)}

    X_train = build_feature_matrix(train_emb, train_categories, cat_to_idx, True)
    X_pool = build_feature_matrix(pool_emb, pool_categories, cat_to_idx, True)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_journals)
    classes = label_encoder.classes_

    clf = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=200, random_state=args.seed)
    clf.fit(X_train, y_train)

    clf_proba = clf.predict_proba(X_pool)

    # Ensemble probability matrix
    print("  Building ensemble probabilities...", file=sys.stderr)
    proba = ensemble_proba_matrix(knn_preds, clf_proba, classes, args.alpha)

    # Apply temperature scaling
    if abs(T - 1.0) > 0.001:
        print(f"  Applying temperature scaling (T={T:.4f})...", file=sys.stderr)
        proba = temperature_scale(proba, T)

    # Build class index
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Load titles
    titles = load_titles(args.dataset, set(pool_dois))

    print("\nReady.", file=sys.stderr)

    def query_journal(query):
        """Process a single journal query."""
        name, count = resolve_journal(query, journal_counts, args.min_papers)
        if name is None:
            if count == 0:
                print(f"\nJournal '{query}' not found or has fewer than "
                      f"{args.min_papers} training papers.")
                # Show closest matches
                matches = [(n, c) for n, c in journal_counts.items()
                           if query.lower() in n.lower()]
                if matches:
                    print("  Possible matches:")
                    for n, c in sorted(matches, key=lambda x: -x[1])[:10]:
                        print(f"    {c:4d} papers  {n}")
            return

        idx = class_to_idx.get(name)
        if idx is None:
            print(f"\nJournal '{name}' not in classifier classes.")
            return

        # Get probabilities for this journal across all pool papers
        journal_probs = proba[:, idx]

        # Sort by probability descending
        ranked_indices = np.argsort(journal_probs)[::-1]

        # Build result list
        papers = []
        for i in ranked_indices:
            doi = pool_dois[i]
            title = titles.get(doi, "(title not available)")
            prob = float(journal_probs[i])
            true_j = pool_journals[i]
            papers.append((doi, title, prob, true_j))

        # Apply threshold and top-k
        filtered = [(doi, title, prob, true_j) for doi, title, prob, true_j in papers
                    if prob >= args.threshold][:args.top_k]

        # Display
        print(f"\n{'=' * 80}")
        print(f"Journal: {name}")
        print(f"Training papers: {count} | Pool papers scoring above "
              f"threshold: {sum(1 for _, _, p, _ in papers if p >= args.threshold)}")
        print(f"{'=' * 80}")

        if not filtered:
            print("No papers above threshold.")
            return

        # Header
        print(f"\n{'#':>3}  {'Prob':>6}  {'Match':>5}  Title")
        print(f"{'':>3}  {'':>6}  {'':>5}  DOI")
        print("-" * 80)

        for rank, (doi, title, prob, true_j) in enumerate(filtered, 1):
            match = "  YES" if true_j == name else ""
            # Truncate title
            title_display = title[:65] + "..." if len(title) > 68 else title
            print(f"{rank:3d}  {prob:5.1%}  {match:>5}  {title_display}")
            print(f"{'':3}  {'':>6}  {'':>5}  {doi}")

        # Summary stats
        n_match = sum(1 for _, _, _, tj in filtered if tj == name)
        if n_match > 0:
            print(f"\n{n_match}/{len(filtered)} shown papers actually published "
                  f"in {name}")

    # Single query or interactive
    if args.journal and not args.interactive:
        query_journal(args.journal)
    elif args.interactive or not args.journal:
        print("\nInteractive mode. Type a journal name (or 'quit' to exit).\n")
        while True:
            try:
                query = input("Journal> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            query_journal(query)


if __name__ == "__main__":
    main()
