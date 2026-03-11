#!/usr/bin/env python3
"""Predict journals for medRxiv preprints with calibrated probabilities.

For each paper, produces a ranked list of candidate journals with
well-calibrated probabilities. Restricts to journals with sufficient
training data (default >=10 papers) and fits temperature scaling on the
validation set for the restricted distribution.

Usage:
  # Predict for a specific paper
  python3 predict_journal.py --doi 10.1101/2021.05.05.21256010

  # Predict for all pool papers, save to file
  python3 predict_journal.py --all --output predictions.json

  # Interactive mode
  python3 predict_journal.py --interactive

  # Just run calibration analysis and save temperature
  python3 predict_journal.py --calibrate-only
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from evaluate_knn import (
    load_embeddings,
    stratified_split_3way,
    cosine_similarity_chunked,
    predict_knn,
)
from train_classifier import build_feature_matrix
from calibrate import ensemble_proba_matrix


def restrict_and_renormalize(proba, eligible_mask):
    """Zero out ineligible journals and renormalise to valid distribution."""
    restricted = proba[:, eligible_mask].copy()
    row_sums = restricted.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-30)
    return restricted / row_sums


def temperature_scale(proba, T):
    """Apply temperature scaling: log, divide by T, re-softmax."""
    log_p = np.log(np.maximum(proba, 1e-30))
    scaled = log_p / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def fit_temperature(proba, true_indices, bounds=(0.1, 20.0)):
    """Find optimal temperature T minimising NLL."""
    def nll(T):
        cal = temperature_scale(proba, T)
        tp = cal[np.arange(len(true_indices)), true_indices]
        return -np.mean(np.log(np.maximum(tp, 1e-30)))

    result = minimize_scalar(nll, bounds=bounds, method='bounded')
    return result.x, result.fun


def reliability_diagram(proba, true_indices, n_bins=15):
    """Top-1 reliability diagram."""
    confidences = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    correct = (predictions == true_indices).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == 0 \
            else (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count > 0:
            mean_conf = float(confidences[mask].mean())
            mean_acc = float(correct[mask].mean())
        else:
            mean_conf = float((lo + hi) / 2)
            mean_acc = 0.0
        bins.append({
            "bin_lower": float(lo), "bin_upper": float(hi),
            "count": count,
            "mean_confidence": mean_conf, "mean_accuracy": mean_acc,
            "gap": abs(mean_acc - mean_conf),
        })
    return bins


def compute_ece(bins):
    """Expected Calibration Error."""
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    return sum(b["count"] * b["gap"] for b in bins) / total


class JournalPredictor:
    """Predict journals with calibrated probabilities."""

    def __init__(self, embeddings_dir="finetuned-specter2/embeddings",
                 dataset_path="labeled_dataset.json",
                 alpha=0.1, k=20, min_papers=10, classifier_C=10.0,
                 seed=42):
        self.min_papers = min_papers
        self.alpha = alpha
        self.seed = seed

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
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.pool_idx = np.concatenate([val_idx, test_idx])

        self.train_journals = [self.journals[i] for i in train_idx]
        self.val_journals = [self.journals[i] for i in val_idx]
        self.test_journals = [self.journals[i] for i in test_idx]
        self.pool_dois = [self.dois[i] for i in self.pool_idx]
        self.pool_journals = [self.journals[i] for i in self.pool_idx]
        self.journal_counts = Counter(self.train_journals)

        # Eligible journals
        self.eligible_journals = sorted(
            [j for j, c in self.journal_counts.items() if c >= min_papers])
        print(f"Eligible journals (>={min_papers} papers): "
              f"{len(self.eligible_journals)}", file=sys.stderr)

        # Train model
        print("Training model...", file=sys.stderr)
        self.train_emb = self.embeddings[train_idx]
        val_emb = self.embeddings[val_idx]
        test_emb = self.embeddings[test_idx]
        pool_emb = self.embeddings[self.pool_idx]

        self.train_categories = [self.categories[i] for i in train_idx]
        val_cats = [self.categories[i] for i in val_idx]
        test_cats = [self.categories[i] for i in test_idx]
        pool_cats = [self.categories[i] for i in self.pool_idx]
        self.unique_cats = sorted(set(self.train_categories))
        self.cat_to_idx = {c: i + 1 for i, c in enumerate(self.unique_cats)}
        self.k = k

        # kNN
        print("  kNN...", file=sys.stderr)
        val_sim = cosine_similarity_chunked(val_emb, self.train_emb)
        knn_preds_val = predict_knn(val_sim, self.train_journals, k=self.k)
        test_sim = cosine_similarity_chunked(test_emb, self.train_emb)
        knn_preds_test = predict_knn(test_sim, self.train_journals, k=self.k)
        pool_sim = cosine_similarity_chunked(pool_emb, self.train_emb)
        knn_preds_pool = predict_knn(pool_sim, self.train_journals, k=self.k)

        # Classifier
        print("  Classifier...", file=sys.stderr)
        X_train = build_feature_matrix(
            self.train_emb, self.train_categories, self.cat_to_idx, True)
        X_val = build_feature_matrix(val_emb, val_cats, self.cat_to_idx, True)
        X_test = build_feature_matrix(
            test_emb, test_cats, self.cat_to_idx, True)
        X_pool = build_feature_matrix(
            pool_emb, pool_cats, self.cat_to_idx, True)

        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(self.train_journals)
        self.all_classes = self.label_encoder.classes_

        self.classifier_C = classifier_C
        self.clf = LogisticRegression(
            C=classifier_C, solver="lbfgs", max_iter=200, random_state=seed)
        self.clf.fit(X_train, y_train)

        # Ensemble probability matrices
        print("  Ensemble probabilities...", file=sys.stderr)
        proba_val_full = ensemble_proba_matrix(
            knn_preds_val, self.clf.predict_proba(X_val),
            self.all_classes, alpha)
        proba_test_full = ensemble_proba_matrix(
            knn_preds_test, self.clf.predict_proba(X_test),
            self.all_classes, alpha)
        self.proba_pool_full = ensemble_proba_matrix(
            knn_preds_pool, self.clf.predict_proba(X_pool),
            self.all_classes, alpha)

        # Restrict to eligible journals
        self.eligible_mask = np.array([
            j in set(self.eligible_journals) for j in self.all_classes])
        self.restricted_classes = self.all_classes[self.eligible_mask]
        self.restricted_class_to_idx = {
            c: i for i, c in enumerate(self.restricted_classes)}

        proba_val = restrict_and_renormalize(proba_val_full, self.eligible_mask)
        proba_test = restrict_and_renormalize(proba_test_full, self.eligible_mask)
        self.proba_pool = restrict_and_renormalize(
            self.proba_pool_full, self.eligible_mask)

        # Val/test true indices in the restricted class set
        val_true = []
        val_valid = []
        for i, j in enumerate(self.val_journals):
            idx = self.restricted_class_to_idx.get(j)
            if idx is not None:
                val_true.append(idx)
                val_valid.append(i)
        val_true = np.array(val_true)
        proba_val_valid = proba_val[val_valid]

        test_true = []
        test_valid = []
        for i, j in enumerate(self.test_journals):
            idx = self.restricted_class_to_idx.get(j)
            if idx is not None:
                test_true.append(idx)
                test_valid.append(i)
        test_true = np.array(test_true)
        proba_test_valid = proba_test[test_valid]

        # --- Calibration ---
        # Step 1: temperature scaling (global adjustment)
        print("  Fitting temperature...", file=sys.stderr)
        self.T, _ = fit_temperature(proba_val_valid, val_true)
        print(f"  T = {self.T:.4f}", file=sys.stderr)

        proba_val_ts = temperature_scale(proba_val_valid, self.T)
        proba_test_ts = temperature_scale(proba_test_valid, self.T)

        # Step 2: isotonic regression on per-class probabilities
        # Fit on val set: collect (predicted_prob, is_correct) for top-k classes
        print("  Fitting isotonic calibration...", file=sys.stderr)
        iso_probs = []
        iso_labels = []
        top_k_cal = 50  # use top-50 predictions per paper for fitting
        for i in range(len(proba_val_ts)):
            top_idx = np.argsort(proba_val_ts[i])[::-1][:top_k_cal]
            for j in top_idx:
                iso_probs.append(proba_val_ts[i, j])
                iso_labels.append(1.0 if j == val_true[i] else 0.0)

        iso_probs = np.array(iso_probs)
        iso_labels = np.array(iso_labels)

        self.iso_reg = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds='clip')
        self.iso_reg.fit(iso_probs, iso_labels)

        proba_val_cal = self._apply_isotonic(proba_val_ts)
        proba_test_cal = self._apply_isotonic(proba_test_ts)

        # Calibrate pool
        proba_pool_ts = temperature_scale(self.proba_pool, self.T)
        self.proba_pool = self._apply_isotonic(proba_pool_ts)

        # Calibration metrics — compare before and after
        bins_val_before = reliability_diagram(proba_val_ts, val_true)
        bins_test_before = reliability_diagram(proba_test_ts, test_true)
        bins_val_after = reliability_diagram(proba_val_cal, val_true)
        bins_test_after = reliability_diagram(proba_test_cal, test_true)

        ece_val_before = compute_ece(bins_val_before)
        ece_test_before = compute_ece(bins_test_before)
        ece_val_after = compute_ece(bins_val_after)
        ece_test_after = compute_ece(bins_test_after)

        val_acc = float((np.argmax(proba_val_cal, axis=1) == val_true).mean())
        test_acc = float((np.argmax(proba_test_cal, axis=1) == test_true).mean())

        self.calibration = {
            "temperature": self.T,
            "n_eligible_journals": len(self.eligible_journals),
            "min_papers": min_papers,
            "val": {
                "n": len(val_true), "top1_accuracy": val_acc,
                "ece_temp_only": ece_val_before,
                "ece_isotonic": ece_val_after,
                "reliability_diagram": bins_val_after,
            },
            "test": {
                "n": len(test_true), "top1_accuracy": test_acc,
                "ece_temp_only": ece_test_before,
                "ece_isotonic": ece_test_after,
                "reliability_diagram": bins_test_after,
            },
        }

        print(f"\n  Calibration (restricted to {len(self.eligible_journals)} "
              f"journals):", file=sys.stderr)
        print(f"    Val:  acc@1={val_acc:.1%}  "
              f"ECE={ece_val_before:.4f} -> {ece_val_after:.4f}  "
              f"n={len(val_true)}", file=sys.stderr)
        print(f"    Test: acc@1={test_acc:.1%}  "
              f"ECE={ece_test_before:.4f} -> {ece_test_after:.4f}  "
              f"n={len(test_true)}", file=sys.stderr)

        print("\n  Reliability (test, after isotonic):", file=sys.stderr)
        for b in bins_test_after:
            if b["count"] > 0:
                print(f"    [{b['bin_lower']:.2f}-{b['bin_upper']:.2f}] "
                      f"n={b['count']:4d}  conf={b['mean_confidence']:.3f}  "
                      f"acc={b['mean_accuracy']:.3f}  "
                      f"gap={b['gap']:.3f}", file=sys.stderr)

        # Load titles
        self.titles = {}
        if Path(dataset_path).exists():
            print("\nLoading titles...", file=sys.stderr)
            with open(dataset_path) as f:
                for paper in json.load(f):
                    doi = paper.get("preprint_doi", "")
                    self.titles[doi] = paper.get("title", "(no title)")

        print("Ready.\n", file=sys.stderr)

    def _apply_isotonic(self, proba_ts):
        """Apply isotonic calibration to probability matrix and renormalise."""
        flat = proba_ts.ravel().copy()
        cal_flat = self.iso_reg.predict(flat)
        cal = cal_flat.reshape(proba_ts.shape)
        row_sums = cal.sum(axis=1, keepdims=True)
        return cal / np.maximum(row_sums, 1e-30)

    def save(self, model_dir="model"):
        """Serialise trained model to disk for inference without retraining."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Classifier and isotonic calibration
        joblib.dump(self.clf, model_dir / "classifier.joblib")
        joblib.dump(self.iso_reg, model_dir / "isotonic.joblib")

        # Training embeddings for kNN
        np.savez_compressed(
            model_dir / "train_embeddings.npz",
            embeddings=self.train_emb)

        # Config: scalars, lists, and mappings
        config = {
            "alpha": self.alpha,
            "k": self.k,
            "T": self.T,
            "min_papers": self.min_papers,
            "classifier_C": self.classifier_C,
            "seed": self.seed,
            "train_journals": self.train_journals,
            "train_categories": self.train_categories,
            "eligible_journals": self.eligible_journals,
            "all_classes": self.all_classes.tolist(),
            "unique_cats": self.unique_cats,
            "cat_to_idx": self.cat_to_idx,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {model_dir}/", file=sys.stderr)

    @classmethod
    def load(cls, model_dir="model", dataset_path="labeled_dataset.json"):
        """Load a saved model for inference, bypassing __init__ training."""
        model_dir = Path(model_dir)

        # Create bare instance without calling __init__
        obj = cls.__new__(cls)

        # Load config
        with open(model_dir / "config.json") as f:
            config = json.load(f)

        obj.alpha = config["alpha"]
        obj.k = config["k"]
        obj.T = config["T"]
        obj.min_papers = config["min_papers"]
        obj.seed = config["seed"]
        obj.train_journals = config["train_journals"]
        obj.eligible_journals = config["eligible_journals"]
        obj.all_classes = np.array(config["all_classes"])
        obj.unique_cats = config["unique_cats"]
        obj.cat_to_idx = config["cat_to_idx"]
        obj.journal_counts = Counter(obj.train_journals)

        # Load classifier and calibration (label_encoder not needed for inference)
        obj.clf = joblib.load(model_dir / "classifier.joblib")
        obj.iso_reg = joblib.load(model_dir / "isotonic.joblib")
        obj.label_encoder = None

        # Load training embeddings
        data = np.load(model_dir / "train_embeddings.npz")
        obj.train_emb = data["embeddings"]

        # Rebuild derived mappings
        obj.eligible_mask = np.array([
            j in set(obj.eligible_journals) for j in obj.all_classes])
        obj.restricted_classes = obj.all_classes[obj.eligible_mask]
        obj.restricted_class_to_idx = {
            c: i for i, c in enumerate(obj.restricted_classes)}

        # Load titles
        obj.titles = {}
        if Path(dataset_path).exists():
            with open(dataset_path) as f:
                for paper in json.load(f):
                    doi = paper.get("preprint_doi", "")
                    obj.titles[doi] = paper.get("title", "(no title)")

        print(f"Model loaded from {model_dir}/  "
              f"({len(obj.eligible_journals)} eligible journals, "
              f"{obj.train_emb.shape[0]} training papers)",
              file=sys.stderr)

        return obj

    def predict_new(self, embeddings, categories, dois=None, titles=None,
                    top_k=10):
        """Score new papers against the saved model.

        Args:
            embeddings: (n_papers, embedding_dim) array
            categories: list of medRxiv category strings
            dois: optional list of DOIs
            titles: optional list of titles

        Returns:
            list of dicts with doi, title, predictions [(journal, prob), ...]
        """
        n = embeddings.shape[0]

        # kNN
        sim = cosine_similarity_chunked(embeddings, self.train_emb)
        knn_preds = predict_knn(sim, self.train_journals, k=self.k)

        # Classifier
        X = build_feature_matrix(
            embeddings, categories, self.cat_to_idx, True)
        clf_proba = self.clf.predict_proba(X)

        # Ensemble
        proba_full = ensemble_proba_matrix(
            knn_preds, clf_proba, self.all_classes, self.alpha)

        # Restrict to eligible journals + renormalise
        proba = restrict_and_renormalize(proba_full, self.eligible_mask)

        # Calibrate: temperature scaling + isotonic
        proba = temperature_scale(proba, self.T)
        proba = self._apply_isotonic(proba)

        # Build results
        top_k = min(top_k, len(self.restricted_classes))
        results = []
        for i in range(n):
            top_indices = np.argsort(proba[i])[::-1][:top_k]
            predictions = [
                (self.restricted_classes[idx], float(proba[i, idx]))
                for idx in top_indices
            ]
            entry = {
                "doi": dois[i] if dois is not None else f"paper_{i}",
                "title": titles[i] if titles is not None else "(no title)",
                "predictions": predictions,
            }
            results.append(entry)

        return results

    def predict(self, doi=None, pool_index=None, top_k=10):
        """Get calibrated journal predictions for a paper.

        Returns list of (journal, probability) tuples, sorted descending.
        """
        if doi is not None:
            # Find in pool
            global_idx = self.doi_to_idx.get(doi)
            if global_idx is None:
                return None, f"DOI not found: {doi}"
            pool_pos = np.where(self.pool_idx == global_idx)[0]
            if len(pool_pos) == 0:
                return None, f"DOI {doi} is in training set, not in prediction pool"
            pool_index = pool_pos[0]
        elif pool_index is None:
            return None, "Specify --doi or pool_index"

        probs = self.proba_pool[pool_index]
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            journal = self.restricted_classes[idx]
            prob = float(probs[idx])
            predictions.append((journal, prob))

        paper_doi = self.pool_dois[pool_index]
        true_journal = self.pool_journals[pool_index]
        title = self.titles.get(paper_doi, "(no title)")

        info = {
            "doi": paper_doi,
            "title": title,
            "true_journal": true_journal,
            "true_journal_eligible": true_journal in self.restricted_class_to_idx,
        }
        return predictions, info

    def predict_all(self, top_k=10):
        """Get predictions for all pool papers."""
        results = []
        for i in range(len(self.pool_idx)):
            predictions, info = self.predict(pool_index=i, top_k=top_k)
            entry = {
                "doi": info["doi"],
                "title": info["title"],
                "true_journal": info["true_journal"],
                "predictions": [
                    {"journal": j, "probability": round(p, 6)}
                    for j, p in predictions
                ],
            }
            # Check if true journal is in top-k
            pred_journals = [j for j, _ in predictions]
            if info["true_journal"] in pred_journals:
                rank = pred_journals.index(info["true_journal"]) + 1
                entry["true_journal_rank"] = rank
                entry["true_journal_probability"] = round(
                    float(dict(predictions)[info["true_journal"]]), 6)
            results.append(entry)
        return results


def display_prediction(predictions, info, show_true=True):
    """Pretty-print a single paper's journal predictions."""
    print(f"\n{'=' * 78}")
    title = info["title"]
    if len(title) > 76:
        title = title[:73] + "..."
    print(f"  {title}")
    print(f"  DOI: {info['doi']}")
    if show_true:
        marker = ""
        if not info["true_journal_eligible"]:
            marker = " (not in eligible set)"
        print(f"  Published in: {info['true_journal']}{marker}")
    print(f"{'=' * 78}")
    print(f"\n  {'#':>3}  {'Prob':>7}  Journal")
    print(f"  {'-' * 60}")

    for rank, (journal, prob) in enumerate(predictions, 1):
        marker = " <--" if journal == info["true_journal"] else ""
        print(f"  {rank:3d}  {prob:6.1%}  {journal}{marker}")

    # Cumulative probability
    cum_prob = sum(p for _, p in predictions)
    print(f"\n  Top-{len(predictions)} cumulative: {cum_prob:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict journals with calibrated probabilities")
    parser.add_argument("--doi", default=None,
                        help="Paper DOI to predict for")
    parser.add_argument("--embeddings-dir", default="finetuned-specter2/embeddings")
    parser.add_argument("--dataset", default="labeled_dataset.json")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top journal predictions (default: 10)")
    parser.add_argument("--min-papers", type=int, default=10,
                        help="Minimum training papers per journal (default: 10)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Ensemble alpha (default: 0.1, tuned for min-papers=10)")
    parser.add_argument("--all", action="store_true",
                        help="Predict for all pool papers and save to --output")
    parser.add_argument("--output", default="journal_predictions.json",
                        help="Output file for --all mode")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Just run calibration and exit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    predictor = JournalPredictor(
        embeddings_dir=args.embeddings_dir,
        dataset_path=args.dataset,
        alpha=args.alpha,
        min_papers=args.min_papers,
        seed=args.seed)

    if args.calibrate_only:
        cal_file = f"calibration_min{args.min_papers}.json"
        with open(cal_file, "w") as f:
            json.dump(predictor.calibration, f, indent=2)
        print(f"Calibration saved to {cal_file}")
        return

    if args.all:
        print(f"Predicting for {len(predictor.pool_idx)} papers...",
              file=sys.stderr)
        results = predictor.predict_all(top_k=args.top_k)

        # Summary stats
        n_eligible = sum(1 for r in results
                         if r["true_journal"] in predictor.restricted_class_to_idx)
        n_in_top1 = sum(1 for r in results if r.get("true_journal_rank") == 1)
        n_in_top10 = sum(1 for r in results
                         if r.get("true_journal_rank", 999) <= 10)

        output = {
            "config": {
                "min_papers": args.min_papers,
                "n_eligible_journals": len(predictor.eligible_journals),
                "temperature": predictor.T,
                "alpha": args.alpha,
                "n_papers": len(results),
                "n_eligible_papers": n_eligible,
            },
            "summary": {
                "top1_accuracy": n_in_top1 / n_eligible if n_eligible else 0,
                "top10_accuracy": n_in_top10 / n_eligible if n_eligible else 0,
            },
            "predictions": results,
        }

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved {len(results)} predictions to {args.output}")
        print(f"Eligible papers: {n_eligible}/{len(results)}")
        print(f"Top-1 accuracy: {n_in_top1}/{n_eligible} "
              f"({n_in_top1/n_eligible:.1%})" if n_eligible else "")
        print(f"Top-10 accuracy: {n_in_top10}/{n_eligible} "
              f"({n_in_top10/n_eligible:.1%})" if n_eligible else "")
        return

    if args.doi:
        predictions, info = predictor.predict(doi=args.doi, top_k=args.top_k)
        if predictions is None:
            print(info)
            return
        display_prediction(predictions, info)
    elif args.interactive:
        print("Interactive mode. Enter a DOI (or 'q' to quit).")
        print("Try: 10.1101/2021.05.05.21256010\n")
        while True:
            try:
                doi = input("DOI> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not doi or doi.lower() in ("q", "quit", "exit"):
                break
            predictions, info = predictor.predict(doi=doi, top_k=args.top_k)
            if predictions is None:
                print(info)
            else:
                display_prediction(predictions, info)
    else:
        # Demo: show predictions for a few random test papers
        rng = np.random.default_rng(args.seed)
        demo_indices = rng.choice(len(predictor.pool_idx), size=5, replace=False)
        for idx in demo_indices:
            predictions, info = predictor.predict(pool_index=idx, top_k=args.top_k)
            display_prediction(predictions, info)


if __name__ == "__main__":
    main()
