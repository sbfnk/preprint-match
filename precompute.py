#!/usr/bin/env python3
"""Precomputation pipeline: full probability matrix for all papers × all journals.

Fetches preprints from medRxiv, embeds them, and computes the complete
probability matrix against all 365 eligible journals. Output is a compact
set of files that the web app can serve instantly.

Usage:
  python3 precompute.py --fetch-only             # Fetch metadata only (no GPU)
  python3 precompute.py --skip-fetch              # Embed + score existing papers
  python3 precompute.py                           # Full run (fetch + embed + score)
  python3 precompute.py --days 365                # Last year only
  python3 precompute.py --all                     # All medRxiv preprints ever
"""

import json
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from extract_labeled_data import fetch_medrxiv_preprints


def fetch_all_papers(start_date, end_date, known_dois):
    """Fetch preprints month by month, filtering known DOIs."""
    all_papers = {}
    cursor = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while cursor < end:
        # Use monthly chunks for efficiency
        chunk_end = min(cursor + timedelta(days=30), end)
        s = cursor.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"  Fetching {s} to {e}...", file=sys.stderr, end=" ")
        raw = fetch_medrxiv_preprints(s, e, max_records=5000)
        added = 0
        for p in raw:
            doi = p.get("doi", "")
            if doi and doi not in known_dois and doi not in all_papers:
                all_papers[doi] = p
                added += 1
        print(f"{len(raw)} fetched, {added} new", file=sys.stderr)
        cursor = chunk_end

    print(f"  Total: {len(all_papers)} new preprints", file=sys.stderr)
    return list(all_papers.values())


def embed_papers(papers, adapter_path="finetuned-specter2/best_adapter"):
    """Embed papers using fine-tuned SPECTER2."""
    from generate_embeddings import (
        load_specter2,
        generate_fulltext_embeddings,
        select_device,
    )

    device = select_device()
    print(f"Loading SPECTER2 on {device}...", file=sys.stderr)
    tokenizer, model = load_specter2(device)

    adapter_path = Path(adapter_path)
    if adapter_path.exists():
        print(f"Loading adapter from {adapter_path}...", file=sys.stderr)
        model.load_adapter(str(adapter_path), set_active=True)

    records = [{
        "title": p.get("title", ""),
        "abstract": p.get("abstract", ""),
        "full_text": p.get("full_text", ""),
    } for p in papers]

    return generate_fulltext_embeddings(
        records, tokenizer, model, device, batch_size=32, stride=256)


def compute_proba_matrix(emb, categories, predictor):
    """Compute full probability matrix: n_papers × n_eligible_journals."""
    from evaluate_knn import cosine_similarity_chunked, predict_knn
    from train_classifier import build_feature_matrix
    from calibrate import ensemble_proba_matrix
    from predict_journal import restrict_and_renormalize, temperature_scale

    sim = cosine_similarity_chunked(emb, predictor.train_emb)
    knn_preds = predict_knn(sim, predictor.train_journals, k=predictor.k)
    X = build_feature_matrix(emb, categories, predictor.cat_to_idx, True)
    clf_proba = predictor.clf.predict_proba(X)
    proba_full = ensemble_proba_matrix(
        knn_preds, clf_proba, predictor.all_classes, predictor.alpha)
    proba = restrict_and_renormalize(proba_full, predictor.eligible_mask)
    proba = temperature_scale(proba, predictor.T)
    proba = predictor._apply_isotonic(proba)
    return proba


def main():
    parser = argparse.ArgumentParser(
        description="Precompute full probability matrix for all journals")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--dataset", default="labeled_dataset.json")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--adapter-path",
                        default="finetuned-specter2/best_adapter")
    parser.add_argument("--days", type=int, default=None,
                        help="Look back N days (default: 365)")
    parser.add_argument("--all", action="store_true",
                        help="Fetch all medRxiv preprints (since June 2019)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Only embed+score existing papers.json")
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch metadata (no GPU needed)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing papers
    papers_path = output_dir / "papers.json"
    emb_path = output_dir / "embeddings.npz"

    if papers_path.exists():
        with open(papers_path) as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} existing papers", file=sys.stderr)
    else:
        papers = []

    existing_dois = {p["doi"] for p in papers}

    # Load embeddings if available
    emb = None
    if emb_path.exists() and not args.fetch_only:
        emb = np.load(emb_path)["embeddings"]

    # ---------- Fetch ----------
    if not args.skip_fetch:
        # Load training DOIs to exclude
        training_dois = set()
        if Path(args.dataset).exists():
            with open(args.dataset) as f:
                for p in json.load(f):
                    training_dois.add(p.get("preprint_doi", ""))
            print(f"Excluding {len(training_dois)} training DOIs",
                  file=sys.stderr)

        known = training_dois | existing_dois

        # Date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        if args.all:
            start_date = "2019-06-01"
        elif args.days:
            start_date = (datetime.now() - timedelta(days=args.days)
                          ).strftime("%Y-%m-%d")
        else:
            start_date = (datetime.now() - timedelta(days=365)
                          ).strftime("%Y-%m-%d")

        print(f"Fetching {start_date} to {end_date}...", file=sys.stderr)
        new_papers = fetch_all_papers(start_date, end_date, known)

        if new_papers:
            for p in new_papers:
                papers.append({
                    "doi": p["doi"],
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                    "category": p.get("category", ""),
                    "date": p.get("date", ""),
                    "authors": p.get("authors", ""),
                    "has_fulltext": bool(p.get("full_text")),
                })

            # Save papers metadata immediately
            with open(papers_path, "w") as f:
                json.dump(papers, f, indent=2)
            print(f"Saved {len(papers)} papers to {papers_path}",
                  file=sys.stderr)
        else:
            print("No new papers found.", file=sys.stderr)

    if args.fetch_only:
        print(f"\nFetch complete: {len(papers)} papers in {papers_path}",
              file=sys.stderr)
        return

    # ---------- Embed ----------
    # Find papers that need embedding
    if emb is not None and emb.shape[0] < len(papers):
        n_existing = emb.shape[0]
        papers_to_embed = papers[n_existing:]
        print(f"Embedding {len(papers_to_embed)} new papers "
              f"({n_existing} already embedded)...", file=sys.stderr)
        new_emb = embed_papers(papers_to_embed, args.adapter_path)
        emb = np.concatenate([emb, new_emb], axis=0)
    elif emb is None:
        print(f"Embedding all {len(papers)} papers...", file=sys.stderr)
        emb = embed_papers(papers, args.adapter_path)
    else:
        print(f"All {len(papers)} papers already embedded.", file=sys.stderr)

    if emb is None or len(papers) == 0:
        print("No papers to score.", file=sys.stderr)
        return

    # Save embeddings
    np.savez_compressed(emb_path, embeddings=emb)

    # ---------- Score ----------
    from predict_journal import JournalPredictor
    predictor = JournalPredictor.load(args.model_dir, args.dataset)

    print(f"Computing {len(papers)} × {len(predictor.restricted_classes)} "
          f"probability matrix...", file=sys.stderr)
    categories = [p.get("category", "") for p in papers]
    proba = compute_proba_matrix(emb, categories, predictor)

    # ---------- Save ----------
    # Papers (already saved during fetch, but save again for consistency)
    with open(papers_path, "w") as f:
        json.dump(papers, f, indent=2)

    # Full probability matrix
    np.savez_compressed(output_dir / "proba_matrix.npz", proba=proba)

    # Journal index
    journals = []
    for j in predictor.restricted_classes:
        journals.append({
            "name": j,
            "training_papers": predictor.journal_counts.get(j, 0),
        })
    with open(output_dir / "journals.json", "w") as f:
        json.dump(journals, f, indent=2)

    # Metadata
    dates = sorted(set(p.get("date", "") for p in papers if p.get("date")))
    meta = {
        "n_papers": len(papers),
        "n_journals": len(journals),
        "date_range": [dates[0], dates[-1]] if dates else [],
        "last_updated": datetime.now().isoformat(),
        "model_dir": args.model_dir,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPrecomputed:", file=sys.stderr)
    print(f"  Papers: {len(papers)}", file=sys.stderr)
    print(f"  Journals: {len(journals)}", file=sys.stderr)
    if dates:
        print(f"  Date range: {dates[0]} to {dates[-1]}", file=sys.stderr)
    print(f"  Matrix: {proba.shape}", file=sys.stderr)
    print(f"  Output: {output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
