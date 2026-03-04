#!/usr/bin/env python3
"""Paper recommendation system for medRxiv preprints.

Two recommendation modes:
1. Journal-based: specify journals you read → find preprints predicted for them
2. Embedding-based: provide example paper DOIs → find similar preprints

Usage:
  # Journal-based: recommend preprints for specific journals
  python3 recommend.py --journals "The Lancet Infectious Diseases" "eLife"
  python3 recommend.py --journals "PLOS ONE" --top-k 30

  # Embedding-based: find preprints similar to given examples
  python3 recommend.py --papers 10.1101/2021.05.05.21256010 10.1101/2022.01.31.22270178

  # Interactive mode
  python3 recommend.py --interactive
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


class RecommendationEngine:
    """Pre-computed recommendation engine for medRxiv preprints."""

    def __init__(self, embeddings_dir="finetuned-specter2/embeddings",
                 dataset_path="labeled_dataset.json",
                 alpha=0.3, k=20, min_papers=5, seed=42):
        self.alpha = alpha
        self.k = k
        self.min_papers = min_papers
        self.seed = seed

        # Load temperature
        cal_path = Path("calibration_results.json")
        if cal_path.exists():
            with open(cal_path) as f:
                self.T = json.load(f).get("temperature", 1.0)
        else:
            self.T = 1.0

        # Load embeddings
        print("Loading embeddings...", file=sys.stderr)
        emb_dir = Path(embeddings_dir)
        self.embeddings, metadata = load_embeddings(emb_dir)
        self.journals = metadata["journals"]
        self.categories = metadata["categories"]
        self.dois = metadata["dois"]
        self.doi_to_idx = {d: i for i, d in enumerate(self.dois)}

        # Split
        train_idx, val_idx, test_idx = stratified_split_3way(
            self.journals, val_size=0.1, test_size=0.2, seed=seed)
        self.train_idx = train_idx
        self.pool_idx = np.concatenate([val_idx, test_idx])

        self.train_journals = [self.journals[i] for i in train_idx]
        self.pool_journals = [self.journals[i] for i in self.pool_idx]
        self.pool_dois = [self.dois[i] for i in self.pool_idx]
        self.pool_categories = [self.categories[i] for i in train_idx]
        self.journal_counts = Counter(self.train_journals)

        # Train model
        print("Training model...", file=sys.stderr)
        train_emb = self.embeddings[train_idx]
        pool_emb = self.embeddings[self.pool_idx]

        # kNN
        pool_sim = cosine_similarity_chunked(pool_emb, train_emb)
        knn_preds = predict_knn(pool_sim, self.train_journals, k=k)

        # Classifier
        train_cats = [self.categories[i] for i in train_idx]
        pool_cats = [self.categories[i] for i in self.pool_idx]
        unique_cats = sorted(set(train_cats))
        cat_to_idx = {c: i + 1 for i, c in enumerate(unique_cats)}

        X_train = build_feature_matrix(train_emb, train_cats, cat_to_idx, True)
        X_pool = build_feature_matrix(pool_emb, pool_cats, cat_to_idx, True)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(self.train_journals)
        self.classes = label_encoder.classes_
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        clf = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=200, random_state=seed)
        clf.fit(X_train, y_train)

        clf_proba = clf.predict_proba(X_pool)

        # Ensemble probability matrix
        self.proba = ensemble_proba_matrix(
            knn_preds, clf_proba, self.classes, alpha)
        if abs(self.T - 1.0) > 0.001:
            self.proba = temperature_scale(self.proba, self.T)

        # Store pool embeddings for similarity search
        self.pool_emb = pool_emb
        self.train_emb = train_emb

        # Load titles
        self.titles = {}
        if Path(dataset_path).exists():
            print("Loading titles...", file=sys.stderr)
            with open(dataset_path) as f:
                dataset = json.load(f)
            for paper in dataset:
                doi = paper.get("preprint_doi", "")
                self.titles[doi] = paper.get("title", "(no title)")

        print(f"Ready. Pool: {len(self.pool_idx)} papers, "
              f"Journals: {len(self.journal_counts)}", file=sys.stderr)

    def recommend_by_journals(self, journal_names, top_k=20, threshold=0.0):
        """Recommend preprints predicted for any of the given journals.

        Aggregates predictions across journals using max probability:
        for each paper, the recommendation score is the max predicted
        probability across the target journals.
        """
        journal_indices = []
        resolved_names = []

        for name in journal_names:
            idx = self.class_to_idx.get(name)
            if idx is not None:
                journal_indices.append(idx)
                resolved_names.append(name)
            else:
                # Try case-insensitive match
                for cls_name in self.classes:
                    if cls_name.lower() == name.lower():
                        idx = self.class_to_idx[cls_name]
                        journal_indices.append(idx)
                        resolved_names.append(cls_name)
                        break
                else:
                    # Substring match
                    matches = [c for c in self.classes
                               if name.lower() in c.lower()]
                    if len(matches) == 1:
                        idx = self.class_to_idx[matches[0]]
                        journal_indices.append(idx)
                        resolved_names.append(matches[0])
                    elif matches:
                        print(f"Ambiguous: '{name}'. Matches: "
                              f"{matches[:5]}", file=sys.stderr)
                    else:
                        print(f"Journal not found: '{name}'", file=sys.stderr)

        if not journal_indices:
            return []

        # For each paper, compute max probability across target journals
        target_probs = self.proba[:, journal_indices]
        max_probs = np.max(target_probs, axis=1)
        best_journal_idx = np.argmax(target_probs, axis=1)

        # Sort by max probability
        ranked = np.argsort(max_probs)[::-1]

        results = []
        for i in ranked:
            prob = float(max_probs[i])
            if prob < threshold:
                break
            if len(results) >= top_k:
                break

            doi = self.pool_dois[i]
            title = self.titles.get(doi, "(no title)")
            predicted_journal = resolved_names[best_journal_idx[i]]
            actual_journal = self.pool_journals[i]

            results.append({
                "rank": len(results) + 1,
                "doi": doi,
                "title": title,
                "probability": prob,
                "predicted_journal": predicted_journal,
                "actual_journal": actual_journal,
                "match": actual_journal in resolved_names,
            })

        return results

    def recommend_by_papers(self, paper_dois, top_k=20):
        """Recommend preprints similar to given example papers.

        Uses cosine similarity in the embedding space, weighted by the
        ensemble's journal prediction confidence for the example papers'
        journals.
        """
        # Find embeddings for the example papers
        example_indices = []
        example_journals = set()
        for doi in paper_dois:
            idx = self.doi_to_idx.get(doi)
            if idx is not None:
                example_indices.append(idx)
                example_journals.add(self.journals[idx])
            else:
                print(f"DOI not found: {doi}", file=sys.stderr)

        if not example_indices:
            return []

        # Compute mean embedding of examples
        example_embs = self.embeddings[example_indices]
        query_emb = example_embs.mean(axis=0, keepdims=True)

        # Cosine similarity to all pool papers
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        pool_norm = self.pool_emb / np.linalg.norm(
            self.pool_emb, axis=1, keepdims=True)
        similarities = (query_norm @ pool_norm.T).squeeze()

        # Optionally boost papers predicted for the same journals
        journal_indices = [self.class_to_idx[j] for j in example_journals
                           if j in self.class_to_idx]
        if journal_indices:
            journal_probs = np.max(self.proba[:, journal_indices], axis=1)
            # Combined score: similarity * (1 + journal_prob)
            scores = similarities * (1.0 + journal_probs)
        else:
            scores = similarities

        # Exclude example papers from results
        example_pool_indices = set()
        for idx in example_indices:
            pool_pos = np.where(self.pool_idx == idx)[0]
            example_pool_indices.update(pool_pos)

        ranked = np.argsort(scores)[::-1]

        results = []
        for i in ranked:
            if int(i) in example_pool_indices:
                continue
            if len(results) >= top_k:
                break

            doi = self.pool_dois[i]
            title = self.titles.get(doi, "(no title)")
            sim = float(similarities[i])
            actual_journal = self.pool_journals[i]

            results.append({
                "rank": len(results) + 1,
                "doi": doi,
                "title": title,
                "similarity": sim,
                "score": float(scores[i]),
                "actual_journal": actual_journal,
            })

        return results


def print_journal_results(results, journal_names):
    """Display journal-based recommendations."""
    if not results:
        print("No results found.")
        return

    print(f"\n{'=' * 80}")
    print(f"Recommendations for: {', '.join(journal_names)}")
    print(f"{'=' * 80}")
    print(f"\n{'#':>3}  {'Prob':>6}  {'Match':>5}  Journal predicted for")
    print(f"{'':>3}  {'':>6}  {'':>5}  Title | DOI")
    print("-" * 80)

    for r in results:
        match = "  YES" if r["match"] else ""
        title = r["title"][:60] + "..." if len(r["title"]) > 63 else r["title"]
        print(f"{r['rank']:3d}  {r['probability']:5.1%}  {match:>5}  "
              f"[{r['predicted_journal']}]")
        print(f"{'':3}  {'':>6}  {'':>5}  {title}")
        print(f"{'':3}  {'':>6}  {'':>5}  {r['doi']}")

    n_match = sum(1 for r in results if r["match"])
    if n_match > 0:
        print(f"\n{n_match}/{len(results)} actually published in target journal(s)")


def print_paper_results(results, example_dois):
    """Display paper-based recommendations."""
    if not results:
        print("No results found.")
        return

    print(f"\n{'=' * 80}")
    print(f"Papers similar to: {', '.join(example_dois[:3])}"
          + (f" (+{len(example_dois) - 3} more)" if len(example_dois) > 3 else ""))
    print(f"{'=' * 80}")
    print(f"\n{'#':>3}  {'Score':>6}  {'Sim':>5}  Title")
    print(f"{'':>3}  {'':>6}  {'':>5}  Journal | DOI")
    print("-" * 80)

    for r in results:
        title = r["title"][:62] + "..." if len(r["title"]) > 65 else r["title"]
        print(f"{r['rank']:3d}  {r['score']:5.3f}  {r['similarity']:5.3f}  "
              f"{title}")
        print(f"{'':3}  {'':>6}  {'':>5}  [{r['actual_journal']}] {r['doi']}")


def main():
    parser = argparse.ArgumentParser(
        description="Recommend medRxiv preprints")
    parser.add_argument("--journals", nargs="+", default=None,
                        help="Target journal names for journal-based recommendation")
    parser.add_argument("--papers", nargs="+", default=None,
                        help="Example paper DOIs for embedding-based recommendation")
    parser.add_argument("--embeddings-dir", default="finetuned-specter2/embeddings")
    parser.add_argument("--dataset", default="labeled_dataset.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--min-papers", type=int, default=5)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    engine = RecommendationEngine(
        embeddings_dir=args.embeddings_dir,
        dataset_path=args.dataset,
        min_papers=args.min_papers)

    if args.journals:
        results = engine.recommend_by_journals(
            args.journals, top_k=args.top_k, threshold=args.threshold)
        print_journal_results(results, args.journals)
    elif args.papers:
        results = engine.recommend_by_papers(
            args.papers, top_k=args.top_k)
        print_paper_results(results, args.papers)
    elif args.interactive:
        print("\nInteractive recommendation mode.")
        print("Commands:")
        print("  j <journal1>, <journal2>, ...  — journal-based")
        print("  p <doi1> <doi2> ...            — paper-based")
        print("  q                              — quit\n")
        while True:
            try:
                line = input("recommend> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() in ("q", "quit", "exit"):
                break
            if line.startswith("j "):
                journals = [j.strip() for j in line[2:].split(",")]
                results = engine.recommend_by_journals(
                    journals, top_k=args.top_k, threshold=args.threshold)
                print_journal_results(results, journals)
            elif line.startswith("p "):
                dois = line[2:].split()
                results = engine.recommend_by_papers(dois, top_k=args.top_k)
                print_paper_results(results, dois)
            else:
                print("Use 'j <journals>' or 'p <dois>'. Type 'q' to quit.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
