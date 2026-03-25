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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from extract_labeled_data import fetch_preprints

# Publishers whose primary purpose is commercial profit
_COMMERCIAL_PUBLISHERS = {
    "AME Publishing Company",
    "Elsevier BV",
    "F1000 Research Ltd",
    "Fortune Journals",
    "Frontiers Media SA",
    "IOP Publishing",
    "Impact Journals, LLC",
    "Informa UK Limited",
    "MDPI AG",
    "Mary Ann Liebert Inc",
    "Ovid Technologies (Wolters Kluwer Health)",
    "SAGE Publications",
    "Springer Science and Business Media LLC",
    "Walter de Gruyter GmbH",
    "Wiley",
}

# Non-profit entities that operate commercially (university presses, etc.)
_MIXED_PUBLISHERS = {
    "Cambridge University Press (CUP)",
    "JMIR Publications Inc.",
    "Oxford University Press (OUP)",
    "PeerJ",
}


def _extract_publishers(dataset_path):
    """Map journal name → most common publisher from labelled data."""
    from collections import Counter
    with open(dataset_path) as f:
        data = json.load(f)
    journal_pubs = {}
    for p in data:
        j, pub = p.get("journal", ""), p.get("publisher", "")
        if j and pub:
            journal_pubs.setdefault(j, []).append(pub)
    return {j: Counter(pubs).most_common(1)[0][0]
            for j, pubs in journal_pubs.items()}


def _classify_publisher(publisher):
    """Classify publisher as commercial, nonprofit, or mixed."""
    if publisher in _COMMERCIAL_PUBLISHERS:
        return "commercial"
    if publisher in _MIXED_PUBLISHERS:
        return "mixed"
    return "nonprofit" if publisher else ""


def _build_month_chunks(start_date, end_date, servers):
    """Build list of (server, start, end) tuples for parallel fetching."""
    chunks = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    for server in servers:
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=30), end)
            chunks.append((server,
                           cursor.strftime("%Y-%m-%d"),
                           chunk_end.strftime("%Y-%m-%d")))
            cursor = chunk_end
    return chunks


def _fetch_chunk(args):
    """Fetch a single month chunk. Used by ThreadPoolExecutor."""
    server, s, e = args
    raw = fetch_preprints(s, e, server, max_records=5000)
    return server, s, e, raw


def fetch_all_papers(start_date, end_date, known_dois,
                     servers=("medrxiv",), existing_papers=None,
                     papers_path=None, workers=10):
    """Fetch preprints in parallel month chunks, filtering known DOIs.

    Saves incrementally to papers_path so progress is not lost if
    interrupted.
    """
    papers_list = list(existing_papers) if existing_papers else []
    seen_dois = {p["doi"] for p in papers_list} | known_dois
    chunks = _build_month_chunks(start_date, end_date, servers)
    total_new = 0

    print(f"  {len(chunks)} month chunks, {workers} parallel workers",
          file=sys.stderr)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_chunk, c): c for c in chunks}
        for future in as_completed(futures):
            server, s, e, raw = future.result()
            added = 0
            for p in raw:
                doi = p.get("doi", "")
                if doi and doi not in seen_dois:
                    seen_dois.add(doi)
                    papers_list.append({
                        "doi": doi,
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                        "category": p.get("category", ""),
                        "date": p.get("date", ""),
                        "authors": p.get("authors", ""),
                        "has_fulltext": bool(p.get("full_text")),
                        "source": server,
                    })
                    added += 1
            total_new += added
            print(f"  {server} {s}→{e}: {len(raw)} fetched, {added} new "
                  f"({total_new} total new)", file=sys.stderr)

            # Save after each completed chunk
            if added > 0 and papers_path:
                with open(papers_path, "w") as f:
                    json.dump(papers_list, f)

    print(f"  Total: {total_new} new preprints, "
          f"{len(papers_list)} papers overall", file=sys.stderr)
    return papers_list


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


def compute_proba_matrix(emb, categories, predictor, chunk_size=2000):
    """Compute full probability matrix: n_papers × n_eligible_journals.

    Processes in chunks to limit memory usage — the full similarity matrix
    (n_papers × n_train) can be 10s of GB and doesn't fit in CI runners.
    """
    from evaluate_knn import predict_knn
    from train_classifier import build_feature_matrix
    from calibrate import ensemble_proba_matrix
    from predict_journal import restrict_and_renormalize, temperature_scale

    n = emb.shape[0]
    n_eligible = int(predictor.eligible_mask.sum())
    proba_all = np.empty((n, n_eligible), dtype=np.float32)

    # Normalise train embeddings once
    train_norm = predictor.train_emb / np.linalg.norm(
        predictor.train_emb, axis=1, keepdims=True)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_emb = emb[start:end]
        chunk_cats = categories[start:end]

        # kNN: compute similarity for this chunk only
        chunk_norm = chunk_emb / np.linalg.norm(
            chunk_emb, axis=1, keepdims=True)
        sim = chunk_norm @ train_norm.T
        knn_preds = predict_knn(sim, predictor.train_journals, k=predictor.k)

        # Classifier (PCA-reduced if available)
        emb_clf = predictor.pca.transform(chunk_emb) if predictor.pca is not None else chunk_emb
        X = build_feature_matrix(
            emb_clf, chunk_cats, predictor.cat_to_idx, True)
        clf_proba = predictor.clf.predict_proba(X)

        # Ensemble + calibrate
        proba_chunk = ensemble_proba_matrix(
            knn_preds, clf_proba, predictor.all_classes, predictor.alpha)
        proba_chunk = restrict_and_renormalize(
            proba_chunk, predictor.eligible_mask)
        proba_chunk = temperature_scale(proba_chunk, predictor.T)
        proba_chunk = predictor._apply_isotonic(proba_chunk)
        proba_all[start:end] = proba_chunk

        print(f"  Scored {end}/{n} papers", file=sys.stderr)

    return proba_all


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
    parser.add_argument("--server", default="both",
                        choices=["medrxiv", "biorxiv", "both"],
                        help="Preprint server(s) to fetch from (default: both)")
    parser.add_argument("--all", action="store_true",
                        help="Fetch all preprints (since June 2019)")
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

        servers = ["medrxiv", "biorxiv"] if args.server == "both" else [args.server]
        print(f"Fetching {start_date} to {end_date} "
              f"({', '.join(servers)})...", file=sys.stderr)
        papers = fetch_all_papers(start_date, end_date, known,
                                  servers=servers,
                                  existing_papers=papers,
                                  papers_path=papers_path)
        print(f"Total: {len(papers)} papers in {papers_path}",
              file=sys.stderr)

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
    # Full probability matrix
    np.savez_compressed(output_dir / "proba_matrix.npz", proba=proba)

    # Journal index (with publisher info from labelled data)
    journal_publisher = _extract_publishers(args.dataset)
    journals = []
    for j in predictor.restricted_classes:
        pub = journal_publisher.get(j, "")
        journals.append({
            "name": j,
            "training_papers": predictor.journal_counts.get(j, 0),
            "publisher": pub,
            "publisher_type": _classify_publisher(pub),
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
