#!/usr/bin/env python3
"""Web app for medRxiv journal predictions.

Serves precomputed predictions: browse journals, see top preprint
candidates, explore individual paper predictions.

Usage:
  python3 webapp.py                          # Development server
  python3 webapp.py --port 8080              # Custom port
  python3 webapp.py --predictions-dir predictions
"""

import html
import json
import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from flask import Flask, render_template, jsonify, request, abort

app = Flask(__name__)

# Global data — loaded once at startup
DATA = {}


def load_data(predictions_dir):
    """Load all precomputed data into memory."""
    d = Path(predictions_dir)

    with open(d / "journals.json") as f:
        DATA["journals"] = json.load(f)
    # Unescape HTML entities in journal names (e.g. &amp; → &)
    for j in DATA["journals"]:
        j["name"] = html.unescape(j["name"])

    with open(d / "papers.json") as f:
        DATA["papers"] = json.load(f)

    with open(d / "meta.json") as f:
        DATA["meta"] = json.load(f)

    # Load full probability matrix for per-paper and per-journal views
    proba_path = d / "proba_matrix.npz"
    if proba_path.exists():
        DATA["proba"] = np.load(proba_path)["proba"]
    else:
        DATA["proba"] = None

    # Build lookup indices
    DATA["paper_by_doi"] = {p["doi"]: i for i, p in enumerate(DATA["papers"])}
    DATA["journal_by_name"] = {
        j["name"]: i for i, j in enumerate(DATA["journals"])
    }

    # Precompute per-journal percentile ranks (across ALL papers)
    # percentile_ranks[i, j] = what percentile paper i is at for journal j
    if DATA["proba"] is not None:
        proba = DATA["proba"]
        # For each journal column, rank papers (higher prob = higher percentile)
        percentile_ranks = np.zeros_like(proba)
        n = proba.shape[0]
        for j in range(proba.shape[1]):
            col = proba[:, j]
            # rankdata: 1 = lowest, n = highest
            order = col.argsort().argsort()  # rank from 0 to n-1
            percentile_ranks[:, j] = (order + 1) / n * 100
        DATA["percentiles"] = percentile_ranks
    else:
        DATA["percentiles"] = None

    # Paper dates as date objects for filtering
    DATA["paper_dates"] = []
    for p in DATA["papers"]:
        try:
            DATA["paper_dates"].append(
                datetime.strptime(p["date"], "%Y-%m-%d").date()
            )
        except (ValueError, KeyError):
            DATA["paper_dates"].append(None)

    # Group journals by first letter for browsing
    letters = {}
    for j in DATA["journals"]:
        first = j["name"][0].upper()
        letters.setdefault(first, []).append(j)
    DATA["journal_letters"] = dict(sorted(letters.items()))


def get_journal_rankings(journal_name, days=None, top_k=20):
    """Compute rankings for a journal from the probability matrix.

    Returns list of paper dicts with probability, percentile, rank.
    Optionally filters to papers from the last N days.
    """
    j_idx = DATA["journal_by_name"].get(journal_name)
    if j_idx is None or DATA["proba"] is None:
        return []

    proba = DATA["proba"]
    percentiles = DATA["percentiles"]
    col = proba[:, j_idx]
    pct_col = percentiles[:, j_idx]

    # Date filter
    if days:
        cutoff = (datetime.now() - timedelta(days=days)).date()
        mask = np.array([
            d is not None and d >= cutoff
            for d in DATA["paper_dates"]
        ])
    else:
        mask = np.ones(len(DATA["papers"]), dtype=bool)

    # Get indices of papers passing the filter, sorted by probability
    filtered_indices = np.where(mask)[0]
    if len(filtered_indices) == 0:
        return []

    filtered_probs = col[filtered_indices]
    ranked = np.argsort(filtered_probs)[::-1][:top_k]

    results = []
    for rank, pos in enumerate(ranked):
        idx = filtered_indices[pos]
        p = DATA["papers"][idx]
        results.append({
            "rank": rank + 1,
            "doi": p["doi"],
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "category": p.get("category", ""),
            "date": p.get("date", ""),
            "authors": p.get("authors", ""),
            "probability": float(col[idx]),
            "percentile": float(pct_col[idx]),
        })

    return results


# ---------- Routes ----------

@app.route("/")
def index():
    """Home page — journal search and stats."""
    return render_template(
        "index.html",
        meta=DATA["meta"],
        n_journals=len(DATA["journals"]),
        journals=DATA["journals"],
        letters=DATA["journal_letters"],
    )


@app.route("/about")
def about():
    """About page."""
    meta = dict(DATA["meta"])
    meta["n_papers_training"] = sum(
        j["training_papers"] for j in DATA["journals"]
    )
    return render_template("about.html", meta=meta)


@app.route("/journal/<path:name>")
def journal_view(name):
    """Journal detail — top predicted preprints."""
    if name not in DATA["journal_by_name"]:
        abort(404)

    days = request.args.get("days", type=int, default=None)
    top_k = request.args.get("top_k", type=int, default=20)

    papers = get_journal_rankings(name, days=days, top_k=top_k)

    journal_info = None
    for j in DATA["journals"]:
        if j["name"] == name:
            journal_info = j
            break

    return render_template(
        "journal.html",
        journal_name=name,
        journal_info=journal_info,
        papers=papers,
        days=days,
        meta=DATA["meta"],
    )


@app.route("/paper/<path:doi>")
def paper_view(doi):
    """Paper detail — predicted journal distribution."""
    idx = DATA["paper_by_doi"].get(doi)
    if idx is None:
        abort(404)

    paper = DATA["papers"][idx]

    # Get journal probabilities and percentiles for this paper
    predictions = []
    if DATA["proba"] is not None:
        row = DATA["proba"][idx]
        pct_row = DATA["percentiles"][idx]
        ranked = np.argsort(row)[::-1]
        for rank, j_idx in enumerate(ranked[:30]):
            predictions.append({
                "journal": DATA["journals"][j_idx]["name"],
                "probability": float(row[j_idx]),
                "percentile": float(pct_row[j_idx]),
                "training_papers": DATA["journals"][j_idx]["training_papers"],
                "rank": rank + 1,
            })

    return render_template(
        "paper.html",
        paper=paper,
        predictions=predictions,
        meta=DATA["meta"],
    )


# ---------- API endpoints ----------

@app.route("/api/search")
def api_search():
    """Unified search: journals + papers (by title or DOI).

    Returns {journals: [...], papers: [...]}.
    Journal priority: name starts with query > word boundary > substring.
    """
    q = request.args.get("q", "").strip()
    q_lower = q.lower()
    if not q:
        return jsonify({"journals": [], "papers": []})

    # --- Journal search ---
    exact = []      # exact match (ignoring "The ")
    prefix = []     # name starts with query
    word_start = []  # query matches start of a word
    substring = []   # query appears anywhere

    q_words = q_lower.split()

    for j in DATA["journals"]:
        name_lower = j["name"].lower()
        name_stripped = (name_lower[4:] if name_lower.startswith("the ")
                         else name_lower)
        if name_stripped == q_lower or name_lower == q_lower:
            exact.append(j)
        elif name_stripped.startswith(q_lower) or name_lower.startswith(q_lower):
            prefix.append(j)
        elif any(w.startswith(q_lower) for w in name_lower.split()):
            word_start.append(j)
        elif q_lower in name_lower:
            substring.append(j)
        elif len(q_words) > 1 and all(w in name_lower for w in q_words):
            substring.append(j)

    for group in (exact, prefix, word_start, substring):
        group.sort(key=lambda x: -x["training_papers"])

    journals = (exact + prefix + word_start + substring)[:15]

    # --- Paper search (by DOI or title) ---
    papers = []
    is_doi = q.startswith("10.") or "doi" in q_lower

    if is_doi:
        # DOI lookup
        for p in DATA["papers"]:
            if q_lower in p["doi"].lower():
                papers.append({
                    "doi": p["doi"],
                    "title": fix_title_filter(p.get("title", "")),
                    "category": p.get("category", ""),
                    "date": p.get("date", ""),
                })
                if len(papers) >= 5:
                    break
    elif len(q) >= 3:
        # Title search — all query words must appear (any order)
        words = q_lower.split()
        for p in DATA["papers"]:
            title_lower = p.get("title", "").lower()
            if all(w in title_lower for w in words):
                papers.append({
                    "doi": p["doi"],
                    "title": fix_title_filter(p.get("title", "")),
                    "category": p.get("category", ""),
                    "date": p.get("date", ""),
                })
                if len(papers) >= 5:
                    break

    return jsonify({"journals": journals, "papers": papers})


@app.route("/api/journal/<path:name>")
def api_journal(name):
    """Get rankings for a journal."""
    if name not in DATA["journal_by_name"]:
        return jsonify({"error": "Journal not found"}), 404
    days = request.args.get("days", type=int, default=None)
    top_k = request.args.get("top_k", type=int, default=20)
    return jsonify(get_journal_rankings(name, days=days, top_k=top_k))


# ---------- Template filters ----------

@app.template_filter("pct")
def pct_filter(value):
    """Format probability as percentage."""
    return f"{value * 100:.1f}%"




_KNOWN_ACRONYMS = {
    "COVID-19", "COVID", "SARS-COV-2", "SARS", "HIV", "AIDS",
    "DNA", "RNA", "PCR", "BMI", "WHO", "UK", "US", "USA", "EU", "ICU",
    "MRI", "CT", "TB", "HPV", "HCV", "HBV", "RSV", "COPD", "PTSD", "RCT",
}


@app.template_filter("fix_title")
def fix_title_filter(title):
    """Fix ALL CAPS titles to title case, preserving known acronyms."""
    if not title:
        return title
    if title != title.upper():
        return title
    # Title case, then restore known acronyms as whole words
    result = title.title()
    for acronym in sorted(_KNOWN_ACRONYMS, key=len, reverse=True):
        pattern = r'\b' + re.escape(acronym.title()) + r'\b'
        result = re.sub(pattern, acronym, result)
    return result


@app.template_filter("top_pct")
def top_pct_filter(percentile):
    """Format percentile as 'Top X%' label."""
    complement = 100 - percentile
    if complement < 0.1:
        return "#1"
    elif complement < 1:
        return f"Top {complement:.1f}%"
    else:
        return f"Top {complement:.0f}%"


@app.template_filter("doi_url")
def doi_url_filter(doi):
    """Convert DOI to medRxiv URL."""
    return f"https://www.medrxiv.org/content/{doi}"


# ---------- Main ----------

# Load data at import time so gunicorn workers have it ready
import os
_predictions_dir = os.environ.get("PREDICTIONS_DIR", "predictions")
load_data(_predictions_dir)
print(f"Loaded {DATA['meta']['n_papers']} papers, "
      f"{DATA['meta']['n_journals']} journals")


def main():
    parser = argparse.ArgumentParser(description="medRxiv predictions web app")
    parser.add_argument("--predictions-dir", default="predictions")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Reload if a different dir was specified
    if args.predictions_dir != _predictions_dir:
        load_data(args.predictions_dir)

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
