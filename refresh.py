#!/usr/bin/env python3
"""Weekly refresh pipeline: score new medRxiv preprints against saved model.

Fetches recent preprints from the medRxiv API, optionally retrieves full
text from the S3 MECA archive, embeds with fine-tuned SPECTER2, and scores
against the saved model. Predictions are stored in predictions/new_papers.json
for downstream tools.

Usage:
  python3 refresh.py                          # Full refresh with S3 sync
  python3 refresh.py --skip-fulltext          # Title+abstract only
  python3 refresh.py --model-dir model        # Custom model directory
  python3 refresh.py --days 14                # Look back 14 days
"""

import json
import argparse
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

from extract_labeled_data import fetch_medrxiv_preprints
from predict_journal import JournalPredictor


# ---------- State management ----------

def load_state(path="refresh_state.json"):
    """Load pipeline state from disk."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "last_refresh_date": None,
        "last_meca_sync": None,
        "processed_dois": [],
    }


def save_state(state, path="refresh_state.json"):
    """Save pipeline state to disk."""
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ---------- Fetch new preprints ----------

def fetch_new_preprints(days, training_dois, processed_dois):
    """Fetch recent preprints from medRxiv API, filtering known DOIs.

    Returns ALL preprints (not just published ones), deduplicated by DOI
    (keeping latest version).
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"Fetching preprints from {start_date} to {end_date}...",
          file=sys.stderr)
    raw = fetch_medrxiv_preprints(start_date, end_date)
    print(f"  Fetched {len(raw)} records from API", file=sys.stderr)

    known_dois = training_dois | processed_dois

    # Deduplicate by DOI (later versions overwrite earlier)
    by_doi = {}
    for p in raw:
        doi = p.get("doi", "")
        if doi and doi not in known_dois:
            by_doi[doi] = p

    papers = list(by_doi.values())
    print(f"  {len(papers)} new preprints after filtering "
          f"({len(known_dois)} known DOIs)", file=sys.stderr)
    return papers


# ---------- Full text from MECA files ----------

def sync_meca_files(meca_dir, last_sync_date=None):
    """Sync new MECA files from S3.

    Uses s5cmd if available, falling back to aws s3 sync.
    Returns the sync timestamp.
    """
    meca_dir = Path(meca_dir)
    meca_dir.mkdir(parents=True, exist_ok=True)

    # Try s5cmd first (faster), fall back to aws cli
    s5cmd = None
    for candidate in ["s5cmd", "./s5cmd"]:
        try:
            subprocess.run([candidate, "version"], capture_output=True,
                           check=True)
            s5cmd = candidate
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    src = "s3://biorxiv-src-monthly/Current_Content/"

    if s5cmd:
        cmd = [s5cmd, "--request-payer", "requester", "sync",
               src, str(meca_dir)]
    else:
        cmd = ["aws", "s3", "sync", "--request-payer", "requester",
               src, str(meca_dir)]

    print(f"Syncing MECA files: {' '.join(cmd)}", file=sys.stderr)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: S3 sync failed: {e}", file=sys.stderr)
        print("Continuing without new full text.", file=sys.stderr)
        return last_sync_date
    except FileNotFoundError:
        print("Warning: Neither s5cmd nor aws CLI found. "
              "Skipping S3 sync.", file=sys.stderr)
        return last_sync_date

    return datetime.now().strftime("%Y-%m-%d")


def extract_fulltext_from_meca(meca_dir, dois):
    """Extract body text from MECA files for given DOIs.

    MECA files are zip archives containing JATS XML in content/*.xml.
    Returns dict mapping DOI to body_text string.
    """
    from parse_xml import parse_jats_xml

    meca_dir = Path(meca_dir)
    if not meca_dir.exists():
        return {}

    # Build a lookup set for speed
    target_dois = set(dois)
    fulltext = {}

    meca_files = list(meca_dir.glob("*.meca"))
    if not meca_files:
        print("  No MECA files found.", file=sys.stderr)
        return {}

    print(f"  Scanning {len(meca_files)} MECA files for {len(target_dois)} "
          f"DOIs...", file=sys.stderr)

    for meca_path in tqdm(meca_files, desc="Extracting full text",
                          file=sys.stderr):
        if not target_dois:
            break  # Found all DOIs

        try:
            with zipfile.ZipFile(meca_path, "r") as zf:
                xml_names = [n for n in zf.namelist()
                             if n.startswith("content/") and n.endswith(".xml")]
                for xml_name in xml_names:
                    with zf.open(xml_name) as xml_file:
                        # Write to temp file for parse_jats_xml
                        with tempfile.NamedTemporaryFile(
                                suffix=".xml", delete=False) as tmp:
                            tmp.write(xml_file.read())
                            tmp_path = Path(tmp.name)

                        try:
                            parsed = parse_jats_xml(tmp_path)
                            doi = parsed.get("doi", "")
                            if doi in target_dois and parsed.get("body_text"):
                                fulltext[doi] = parsed["body_text"]
                                target_dois.discard(doi)
                        finally:
                            tmp_path.unlink(missing_ok=True)

        except (zipfile.BadZipFile, Exception) as e:
            # Skip corrupt MECA files
            continue

    print(f"  Found full text for {len(fulltext)}/{len(dois)} papers",
          file=sys.stderr)
    return fulltext


# ---------- Embedding ----------

def embed_papers(papers, adapter_path="finetuned-specter2/best_adapter"):
    """Embed papers using fine-tuned SPECTER2.

    Papers with full_text get full-text embedding; others get
    title+abstract.
    """
    from generate_embeddings import (
        load_specter2,
        generate_fulltext_embeddings,
        select_device,
    )

    device = select_device()
    print(f"Loading SPECTER2 model on {device}...", file=sys.stderr)
    tokenizer, model = load_specter2(device)

    # Load fine-tuned adapter if available
    adapter_path = Path(adapter_path)
    if adapter_path.exists():
        print(f"Loading fine-tuned adapter from {adapter_path}...",
              file=sys.stderr)
        model.load_adapter(str(adapter_path), set_active=True)
    else:
        print(f"Warning: adapter not found at {adapter_path}, "
              f"using base SPECTER2", file=sys.stderr)

    # Build records in the format generate_fulltext_embeddings expects
    records = []
    for p in papers:
        records.append({
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "full_text": p.get("full_text", ""),
        })

    print(f"Embedding {len(records)} papers...", file=sys.stderr)
    embeddings = generate_fulltext_embeddings(
        records, tokenizer, model, device, batch_size=32, stride=256)

    return embeddings


# ---------- Existing predictions management ----------

def load_existing_predictions(output_dir):
    """Load existing prediction store, returning papers list and DOI set."""
    pred_path = Path(output_dir) / "new_papers.json"
    if pred_path.exists():
        with open(pred_path) as f:
            papers = json.load(f)
        doi_set = {p["doi"] for p in papers}
        return papers, doi_set
    return [], set()


def save_predictions(papers, embeddings, output_dir):
    """Save predictions and embeddings to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "new_papers.json", "w") as f:
        json.dump(papers, f, indent=2)

    np.savez_compressed(output_dir / "new_embeddings.npz",
                        embeddings=embeddings)

    print(f"Saved {len(papers)} predictions to {output_dir}/",
          file=sys.stderr)


# ---------- Re-embed papers that now have full text ----------

def check_fulltext_updates(existing_papers, fulltext_map):
    """Find previously scored papers that now have full text available.

    Returns list of indices into existing_papers that need re-embedding.
    """
    update_indices = []
    for i, p in enumerate(existing_papers):
        if not p.get("has_fulltext") and p["doi"] in fulltext_map:
            update_indices.append(i)
    return update_indices


# ---------- Main pipeline ----------

def main():
    parser = argparse.ArgumentParser(
        description="Weekly refresh: score new medRxiv preprints")
    parser.add_argument("--model-dir", default="model",
                        help="Saved model directory (default: model)")
    parser.add_argument("--dataset", default="labeled_dataset.json",
                        help="Training dataset for titles/DOIs")
    parser.add_argument("--output-dir", default="predictions",
                        help="Output directory for predictions")
    parser.add_argument("--state-file", default="refresh_state.json",
                        help="Pipeline state file")
    parser.add_argument("--skip-fulltext", action="store_true",
                        help="Skip S3 sync and full-text extraction")
    parser.add_argument("--meca-dir", default="meca",
                        help="Directory for MECA files")
    parser.add_argument("--adapter-path",
                        default="finetuned-specter2/best_adapter",
                        help="Path to fine-tuned adapter")
    parser.add_argument("--days", type=int, default=30,
                        help="Look back N days for new preprints (default: 30)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of journal predictions per paper")
    args = parser.parse_args()

    # Load state
    state = load_state(args.state_file)
    processed_dois = set(state.get("processed_dois", []))

    # Load training DOIs from dataset
    training_dois = set()
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        with open(dataset_path) as f:
            for paper in json.load(f):
                training_dois.add(paper.get("preprint_doi", ""))

    # Load existing predictions
    existing_papers, existing_dois = load_existing_predictions(args.output_dir)

    # Fetch new preprints
    all_known = training_dois | processed_dois | existing_dois
    papers = fetch_new_preprints(args.days, training_dois,
                                 processed_dois | existing_dois)

    if not papers and not args.skip_fulltext:
        # Even if no new papers, check for full-text updates
        pass

    # Full text extraction
    fulltext_map = {}
    if not args.skip_fulltext:
        # Sync MECA files
        sync_date = sync_meca_files(
            args.meca_dir, state.get("last_meca_sync"))
        state["last_meca_sync"] = sync_date

        # Extract full text for new papers + papers missing full text
        all_dois_needing_text = [p["doi"] for p in papers]
        for ep in existing_papers:
            if not ep.get("has_fulltext"):
                all_dois_needing_text.append(ep["doi"])

        if all_dois_needing_text:
            fulltext_map = extract_fulltext_from_meca(
                args.meca_dir, all_dois_needing_text)

    # Attach full text to new papers
    for p in papers:
        doi = p["doi"]
        if doi in fulltext_map:
            p["full_text"] = fulltext_map[doi]
            p["has_fulltext"] = True
        else:
            p["full_text"] = ""
            p["has_fulltext"] = False

    # Check for existing papers that now have full text
    update_indices = check_fulltext_updates(existing_papers, fulltext_map)
    if update_indices:
        print(f"\n{len(update_indices)} existing papers now have full text, "
              f"will re-embed and re-score.", file=sys.stderr)

    # Combine: new papers + papers needing re-embedding
    papers_to_embed = papers.copy()
    for idx in update_indices:
        ep = existing_papers[idx]
        papers_to_embed.append({
            "doi": ep["doi"],
            "title": ep.get("title", ""),
            "abstract": ep.get("abstract", ""),
            "category": ep.get("category", ""),
            "date": ep.get("date", ""),
            "full_text": fulltext_map[ep["doi"]],
            "has_fulltext": True,
        })

    if not papers_to_embed:
        print("\nNo new papers to process. Pipeline is up to date.",
              file=sys.stderr)
        save_state(state, args.state_file)
        return

    # Embed
    print(f"\nEmbedding {len(papers_to_embed)} papers...", file=sys.stderr)
    embeddings = embed_papers(papers_to_embed, args.adapter_path)

    # Score against saved model
    print("\nScoring against saved model...", file=sys.stderr)
    predictor = JournalPredictor.load(args.model_dir, args.dataset)

    categories = [p.get("category", "") for p in papers_to_embed]
    dois = [p["doi"] for p in papers_to_embed]
    titles = [p.get("title", "(no title)") for p in papers_to_embed]

    results = predictor.predict_new(
        embeddings, categories, dois, titles, top_k=args.top_k)

    # Split results: new papers vs re-scored existing papers
    n_new = len(papers)
    new_results = results[:n_new]
    updated_results = results[n_new:]

    # Build prediction entries for new papers
    refresh_date = datetime.now().strftime("%Y-%m-%d")
    new_entries = []
    for i, (paper, result) in enumerate(zip(papers, new_results)):
        entry = {
            "doi": paper["doi"],
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "category": paper.get("category", ""),
            "date": paper.get("date", ""),
            "has_fulltext": paper.get("has_fulltext", False),
            "predictions": [
                {"journal": j, "probability": round(p, 6)}
                for j, p in result["predictions"]
            ],
            "refresh_date": refresh_date,
        }
        new_entries.append(entry)

    # Update existing papers that got re-embedded
    updated_dois = set()
    for idx, result in zip(update_indices, updated_results):
        existing_papers[idx]["has_fulltext"] = True
        existing_papers[idx]["predictions"] = [
            {"journal": j, "probability": round(p, 6)}
            for j, p in result["predictions"]
        ]
        existing_papers[idx]["refresh_date"] = refresh_date
        updated_dois.add(existing_papers[idx]["doi"])

    # Merge: existing papers (with updates) + new papers
    all_papers = existing_papers + new_entries

    # Build combined embeddings array
    emb_path = Path(args.output_dir) / "new_embeddings.npz"
    if emb_path.exists():
        old_emb = np.load(emb_path)["embeddings"]
        # Replace embeddings for updated papers
        if update_indices:
            new_emb_for_updates = embeddings[n_new:]
            for local_idx, global_idx in enumerate(update_indices):
                old_emb[global_idx] = new_emb_for_updates[local_idx]
        # Append new paper embeddings
        new_emb = embeddings[:n_new]
        all_emb = np.concatenate([old_emb, new_emb], axis=0)
    else:
        all_emb = embeddings[:n_new]

    # Save
    save_predictions(all_papers, all_emb, args.output_dir)

    # Update state
    new_dois = [p["doi"] for p in papers]
    state["processed_dois"] = list(processed_dois | set(new_dois))
    state["last_refresh_date"] = refresh_date
    save_state(state, args.state_file)

    # Summary
    print(f"\nRefresh complete:", file=sys.stderr)
    print(f"  New papers scored: {len(new_entries)}", file=sys.stderr)
    print(f"  Papers re-scored (full text now available): "
          f"{len(update_indices)}", file=sys.stderr)
    print(f"  Total papers in store: {len(all_papers)}", file=sys.stderr)

    # Show a few example predictions
    if new_entries:
        print(f"\nSample predictions:", file=sys.stderr)
        for entry in new_entries[:3]:
            title = entry["title"][:60] + "..." if len(
                entry["title"]) > 60 else entry["title"]
            top = entry["predictions"][0]
            print(f"  {entry['doi']}: {title}", file=sys.stderr)
            print(f"    Top: {top['journal']} ({top['probability']:.1%})",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
