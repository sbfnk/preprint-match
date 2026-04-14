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
import os
import re
import sqlite3
import sys
import threading
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

import numpy as np
from flask import Flask, render_template, jsonify, request, abort

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600  # cache static files 1 hour

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

    # Load community evaluation data (Sciety, PCI, PubPeer)
    reviews_path = d / "community_reviews.json"
    if not reviews_path.exists():
        reviews_path = d / "sciety_reviews.json"  # fallback
    if reviews_path.exists():
        with open(reviews_path) as f:
            DATA["reviews"] = json.load(f)
        print(f"Loaded {len(DATA['reviews'])} community-evaluated papers")
    else:
        DATA["reviews"] = {}

    # Load full probability matrix for per-paper and per-journal views
    proba_path = d / "proba_matrix.npz"
    if proba_path.exists():
        DATA["proba"] = np.load(proba_path)["proba"].astype(np.float32)
    else:
        DATA["proba"] = None

    # Build lookup indices
    DATA["paper_by_doi"] = {p["doi"]: i for i, p in enumerate(DATA["papers"])}
    DATA["journal_by_name"] = {
        j["name"]: i for i, j in enumerate(DATA["journals"])
    }

    # Precompute sorted probability columns for fast percentile lookups
    if DATA["proba"] is not None:
        DATA["proba_sorted"] = np.sort(DATA["proba"], axis=0)
        DATA["proba_mean"] = DATA["proba"].mean(axis=0)
    else:
        DATA["proba_sorted"] = None
        DATA["proba_mean"] = None

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

    # Load training data for search and ground-truth display
    dataset_path = os.environ.get("TRAINING_DATASET", "labeled_dataset.json")
    if Path(dataset_path).exists():
        with open(dataset_path) as f:
            training = json.load(f)
        # Map preprint DOI → actual journal for ground truth
        DATA["true_journal"] = {
            p["preprint_doi"]: p["journal"] for p in training
            if p.get("preprint_doi") and p.get("journal")
        }
        # Add training papers that aren't already in predictions
        existing_dois = DATA["paper_by_doi"]
        DATA["training_papers"] = []
        for p in training:
            doi = p.get("preprint_doi", "")
            if doi and doi not in existing_dois:
                DATA["training_papers"].append({
                    "doi": doi,
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                    "category": p.get("category", ""),
                    "date": p.get("date", ""),
                    "authors": p.get("authors", ""),
                    "journal": p.get("journal", ""),
                    "source": p.get("source", "medrxiv"),
                })
        # Index training papers by DOI
        DATA["training_by_doi"] = {
            p["doi"]: p for p in DATA["training_papers"]
        }
        print(f"Loaded {len(DATA['true_journal'])} ground-truth labels, "
              f"{len(DATA['training_papers'])} training-only papers")
    else:
        DATA["true_journal"] = {}
        DATA["training_papers"] = []
        DATA["training_by_doi"] = {}

    # Pre-build search index: combined list with lowercased titles, sorted
    # newest first, to avoid rebuilding on every search request
    all_papers = list(DATA["papers"]) + DATA["training_papers"]
    search_index = []
    for p in all_papers:
        title = p.get("title", "")
        authors = p.get("authors", "")
        search_index.append({
            "doi": p.get("doi", ""),
            "doi_lower": p.get("doi", "").lower(),
            "title": title,
            "authors": authors,
            "search_text": (title + " " + authors).lower(),
            "category": p.get("category", ""),
            "date": p.get("date", ""),
            "journal": p.get("journal") or DATA["true_journal"].get(
                p.get("doi", "")),
            "source": p.get("source", "medrxiv"),
        })
    search_index.sort(key=lambda x: x["date"], reverse=True)
    DATA["search_index"] = search_index
    print(f"Search index: {len(search_index)} papers")


def percentile(prob_value, j_idx):
    """Compute percentile rank of a probability value for a journal column.

    Uses binary search on the pre-sorted column — O(log n) per call.
    """
    sorted_col = DATA["proba_sorted"][:, j_idx]
    rank = np.searchsorted(sorted_col, prob_value, side="right")
    return rank / len(sorted_col) * 100


def get_journal_rankings(journal_name, days=None, top_k=20):
    """Compute rankings for a journal from the probability matrix.

    Returns list of paper dicts with probability, percentile, rank.
    Optionally filters to papers from the last N days.
    """
    j_idx = DATA["journal_by_name"].get(journal_name)
    if j_idx is None or DATA["proba"] is None:
        return []

    proba = DATA["proba"]
    col = proba[:, j_idx]

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

    baseline = float(DATA["proba_mean"][j_idx]) if DATA["proba_mean"] is not None else 0

    results = []
    for rank, pos in enumerate(ranked):
        idx = filtered_indices[pos]
        p = DATA["papers"][idx]
        prob = float(col[idx])
        results.append({
            "rank": rank + 1,
            "doi": p["doi"],
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "category": p.get("category", ""),
            "date": p.get("date", ""),
            "authors": p.get("authors", ""),
            "probability": prob,
            "percentile": float(percentile(col[idx], j_idx)),
            "lift": prob / baseline if baseline > 0 else None,
        })

    return results


# ---------- Analytics ----------

ANALYTICS_DB = os.environ.get("ANALYTICS_DB", "analytics.db")
STATS_PASSWORD = os.environ.get("STATS_PASSWORD", "")


def get_analytics_db():
    """Get thread-local SQLite connection."""
    t = threading.current_thread()
    if not hasattr(t, "_analytics_db"):
        conn = sqlite3.connect(ANALYTICS_DB, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        t._analytics_db = conn
    return t._analytics_db


def init_analytics_db():
    """Create the hits table if it doesn't exist."""
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            path TEXT NOT NULL,
            referrer TEXT,
            region TEXT,
            device TEXT,
            browser TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hits_ts ON hits(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hits_path ON hits(path)")
    conn.commit()
    conn.close()


def parse_ua(ua_string):
    """Extract device type and browser family from User-Agent."""
    ua = (ua_string or "").lower()
    if "bot" in ua or "crawl" in ua or "spider" in ua:
        device = "bot"
    elif "mobile" in ua or "android" in ua:
        device = "mobile"
    elif "tablet" in ua or "ipad" in ua:
        device = "tablet"
    else:
        device = "desktop"
    if "firefox" in ua:
        browser = "Firefox"
    elif "edg" in ua:
        browser = "Edge"
    elif "chrome" in ua or "chromium" in ua:
        browser = "Chrome"
    elif "safari" in ua:
        browser = "Safari"
    else:
        browser = "Other"
    return device, browser


def require_stats_auth(f):
    """Password check via query param or basic auth."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not STATS_PASSWORD:
            abort(403)
        if request.args.get("key") == STATS_PASSWORD:
            return f(*args, **kwargs)
        auth = request.authorization
        if auth and auth.password == STATS_PASSWORD:
            return f(*args, **kwargs)
        return ("Unauthorised", 401,
                {"WWW-Authenticate": 'Basic realm="Stats"'})
    return decorated


init_analytics_db()


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
    top_k = min(request.args.get("top_k", type=int, default=20), 200)

    papers = get_journal_rankings(name, days=days, top_k=top_k)

    j_idx = DATA["journal_by_name"][name]
    journal_info = DATA["journals"][j_idx]

    # Average probability across all preprints for this journal (baseline rate)
    baseline_pct = None
    if DATA["proba_mean"] is not None:
        baseline_pct = float(DATA["proba_mean"][j_idx]) * 100

    return render_template(
        "journal.html",
        journal_name=name,
        journal_info=journal_info,
        papers=papers,
        days=days,
        baseline_pct=baseline_pct,
        meta=DATA["meta"],
        reviews=DATA["reviews"],
    )


@app.route("/paper/<path:doi>")
def paper_view(doi):
    """Paper detail — predicted journal distribution."""
    idx = DATA["paper_by_doi"].get(doi)
    training_paper = DATA["training_by_doi"].get(doi)

    if idx is None and training_paper is None:
        abort(404)

    # Use prediction paper if available, otherwise training paper
    if idx is not None:
        paper = DATA["papers"][idx]
    else:
        paper = training_paper

    # Ground truth: did this paper end up in a known journal?
    true_journal = DATA["true_journal"].get(doi)
    is_training = doi in DATA["true_journal"]

    # Get journal probabilities and percentiles for this paper
    predictions = []
    prediction_set_size = 0
    true_journal_rank = None
    if idx is not None and DATA["proba"] is not None:
        row = DATA["proba"][idx]
        ranked = np.argsort(row)[::-1]

        # Compute 50% prediction set
        cumsum = 0.0
        coverage_target = 0.50
        for j_idx in ranked:
            cumsum += float(row[j_idx])
            prediction_set_size += 1
            if cumsum >= coverage_target:
                break

        for rank, j_idx in enumerate(ranked[:30]):
            j = DATA["journals"][j_idx]
            prob = float(row[j_idx])
            baseline = float(DATA["proba_mean"][j_idx]) if DATA["proba_mean"] is not None else 0
            is_true = (true_journal and j["name"] == true_journal)
            if is_true:
                true_journal_rank = rank + 1
            predictions.append({
                "journal": j["name"],
                "probability": prob,
                "percentile": float(percentile(row[j_idx], j_idx)),
                "training_papers": j["training_papers"],
                "publisher": j.get("publisher", ""),
                "publisher_type": j.get("publisher_type", ""),
                "rank": rank + 1,
                "lift": prob / baseline if baseline > 0 else None,
                "in_prediction_set": rank < prediction_set_size,
                "is_true": is_true,
            })

    return render_template(
        "paper.html",
        paper=paper,
        predictions=predictions,
        prediction_set_size=prediction_set_size,
        true_journal=true_journal,
        true_journal_rank=true_journal_rank,
        true_journal_in_set=true_journal in DATA["journal_by_name"] if true_journal else False,
        is_training=is_training,
        meta=DATA["meta"],
        reviews=DATA["reviews"].get(doi),
    )


# ---------- Feed routes ----------

def get_feed_rankings(journal_names, days=None, top_k=50, keywords=None):
    """Compute multi-journal feed from the probability matrix.

    For each paper, takes the max probability across selected journals.
    Optionally filters by keywords (all must match title or abstract).
    Returns ranked papers and the list of resolved journal names.
    """
    journal_indices = []
    resolved = []
    for name in journal_names:
        idx = DATA["journal_by_name"].get(name)
        if idx is not None:
            journal_indices.append(idx)
            resolved.append(name)

    if not journal_indices or DATA["proba"] is None:
        return [], resolved

    proba = DATA["proba"]
    target_probs = proba[:, journal_indices]
    max_probs = np.max(target_probs, axis=1)
    best_journal_local = np.argmax(target_probs, axis=1)

    # Date filter
    if days:
        cutoff = (datetime.now() - timedelta(days=days)).date()
        mask = np.array([
            d is not None and d >= cutoff
            for d in DATA["paper_dates"]
        ])
    else:
        mask = np.ones(len(DATA["papers"]), dtype=bool)

    filtered = np.where(mask)[0]
    if len(filtered) == 0:
        return [], resolved

    filtered_probs = max_probs[filtered]
    ranked = np.argsort(filtered_probs)[::-1]

    # Keyword filter: all words must appear in title or abstract
    kw_lower = [w.lower() for w in keywords] if keywords else []

    results = []
    for pos in ranked:
        if len(results) >= top_k:
            break
        idx = filtered[pos]
        p = DATA["papers"][idx]

        if kw_lower:
            text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
            if not all(w in text for w in kw_lower):
                continue

        prob = float(max_probs[idx])
        best_j = resolved[int(best_journal_local[idx])]
        results.append({
            "rank": len(results) + 1,
            "doi": p["doi"],
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "category": p.get("category", ""),
            "date": p.get("date", ""),
            "authors": p.get("authors", ""),
            "probability": prob,
            "matched_journal": best_j,
            "source": p.get("source", "medrxiv"),
        })

    return results, resolved


@app.route("/feed")
def feed_view():
    """Custom feed — ranked preprints across user-selected journals."""
    journal_names = request.args.getlist("j")
    if not journal_names:
        from flask import redirect
        return redirect("/")

    days = request.args.get("days", type=int, default=30)
    top_k = min(request.args.get("top_k", type=int, default=50), 200)
    query = request.args.get("q", "").strip()
    keywords = query.split() if query else None
    papers, resolved = get_feed_rankings(
        journal_names, days=days, top_k=top_k, keywords=keywords)

    # Pre-build query params for URL construction in template
    from urllib.parse import quote
    journal_params = "&".join(f"j={quote(n)}" for n in resolved)

    return render_template(
        "feed.html",
        papers=papers,
        journal_names=resolved,
        journal_params=journal_params,
        query=query,
        days=days,
        meta=DATA["meta"],
        reviews=DATA["reviews"],
    )


@app.route("/feed.rss")
def feed_rss():
    """RSS feed for a custom journal selection."""
    journal_names = request.args.getlist("j")
    if not journal_names:
        abort(400)

    days = request.args.get("days", type=int, default=30)
    top_k = min(request.args.get("top_k", type=int, default=50), 200)
    query = request.args.get("q", "").strip()
    keywords = query.split() if query else None
    papers, resolved = get_feed_rankings(
        journal_names, days=days, top_k=top_k, keywords=keywords)

    from urllib.parse import quote
    journal_params = "&amp;".join(f"j={quote(n)}" for n in resolved)
    if query:
        journal_params += f"&amp;q={quote(query)}"

    from flask import make_response
    resp = make_response(render_template(
        "feed.xml",
        papers=papers,
        journal_names=resolved,
        journal_params=journal_params,
        days=days,
        query=query,
        feed_url=request.url,
        site_url=request.host_url.rstrip("/"),
    ))
    resp.headers["Content-Type"] = "application/rss+xml; charset=utf-8"
    return resp


@app.route("/api/feed")
def api_feed():
    """JSON API for feed results."""
    journal_names = request.args.getlist("j")
    if not journal_names:
        return jsonify({"papers": [], "journals": []})

    days = request.args.get("days", type=int, default=30)
    top_k = min(request.args.get("top_k", type=int, default=50), 200)
    query = request.args.get("q", "").strip()
    keywords = query.split() if query else None
    papers, resolved = get_feed_rankings(
        journal_names, days=days, top_k=top_k, keywords=keywords)

    # Add review info
    for p in papers:
        r = DATA["reviews"].get(p["doi"])
        if r:
            p["review_url"] = list(r["urls"].values())[0]
            p["review_sources"] = r["sources"]

    return jsonify({"papers": papers, "journals": resolved})


# ---------- Analytics routes ----------

@app.route("/hit", methods=["POST"])
def hit():
    """Record a page view via navigator.sendBeacon."""
    data = request.get_json(silent=True, force=True) or {}
    path = (data.get("p") or "")[:500]
    referrer = (data.get("r") or "")[:500] or None
    if referrer and "preprints.epiforecasts.io" in referrer:
        referrer = None

    region = request.headers.get("Fly-Region", "unknown")
    device, browser = parse_ua(request.headers.get("User-Agent", ""))
    if device == "bot":
        return "", 204

    try:
        conn = get_analytics_db()
        conn.execute(
            "INSERT INTO hits (timestamp, path, referrer, region, device, browser) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
             path, referrer, region, device, browser),
        )
        conn.commit()
    except Exception:
        pass
    return "", 204


@app.route("/stats")
@require_stats_auth
def stats():
    """Analytics dashboard."""
    days = request.args.get("days", type=int, default=30)
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    conn = get_analytics_db()
    conn.row_factory = sqlite3.Row

    total = conn.execute(
        "SELECT COUNT(*) FROM hits WHERE timestamp >= ?", (cutoff,)
    ).fetchone()[0]

    daily = conn.execute(
        "SELECT DATE(timestamp) as day, COUNT(*) as count "
        "FROM hits WHERE timestamp >= ? GROUP BY day ORDER BY day",
        (cutoff,),
    ).fetchall()

    pages = conn.execute(
        "SELECT path, COUNT(*) as count "
        "FROM hits WHERE timestamp >= ? "
        "GROUP BY path ORDER BY count DESC LIMIT 30",
        (cutoff,),
    ).fetchall()

    referrers = conn.execute(
        "SELECT referrer, COUNT(*) as count "
        "FROM hits WHERE timestamp >= ? AND referrer IS NOT NULL "
        "GROUP BY referrer ORDER BY count DESC LIMIT 15",
        (cutoff,),
    ).fetchall()

    devices = conn.execute(
        "SELECT device, COUNT(*) as count "
        "FROM hits WHERE timestamp >= ? GROUP BY device ORDER BY count DESC",
        (cutoff,),
    ).fetchall()

    browsers = conn.execute(
        "SELECT browser, COUNT(*) as count "
        "FROM hits WHERE timestamp >= ? GROUP BY browser ORDER BY count DESC",
        (cutoff,),
    ).fetchall()

    regions = conn.execute(
        "SELECT region, COUNT(*) as count "
        "FROM hits WHERE timestamp >= ? "
        "GROUP BY region ORDER BY count DESC LIMIT 20",
        (cutoff,),
    ).fetchall()

    conn.row_factory = None

    return render_template(
        "stats.html",
        total=total,
        daily=daily,
        pages=pages,
        referrers=referrers,
        devices=devices,
        browsers=browsers,
        regions=regions,
        days=days,
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
    # Strip DOI URL prefixes so users can paste full URLs
    for prefix in ("https://doi.org/", "http://doi.org/",
                    "https://www.medrxiv.org/content/",
                    "http://www.medrxiv.org/content/",
                    "https://www.biorxiv.org/content/",
                    "http://www.biorxiv.org/content/"):
        if q.lower().startswith(prefix):
            q = q[len(prefix):]
            break
    # Strip trailing version (e.g., "v1", "v2")
    q = re.sub(r'v\d+$', '', q).rstrip('/')
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
    # Uses pre-built search index (already sorted newest first)
    papers = []
    is_doi = q.startswith("10.") or q_lower.startswith("doi:")
    limit = 10

    if is_doi:
        for p in DATA["search_index"]:
            if q_lower in p["doi_lower"]:
                papers.append({
                    "doi": p["doi"],
                    "title": fix_title_filter(p["title"]),
                    "authors": p["authors"],
                    "category": p["category"],
                    "date": p["date"],
                    "journal": p["journal"],
                    "source": p["source"],
                })
                if len(papers) >= limit:
                    break
    elif len(q) >= 3:
        words = q_lower.split()
        for p in DATA["search_index"]:
            if all(w in p["search_text"] for w in words):
                papers.append({
                    "doi": p["doi"],
                    "title": fix_title_filter(p["title"]),
                    "authors": p["authors"],
                    "category": p["category"],
                    "date": p["date"],
                    "journal": p["journal"],
                    "source": p["source"],
                })
                if len(papers) >= limit:
                    break

    return jsonify({"journals": journals, "papers": papers})


@app.route("/api/journal/<path:name>")
def api_journal(name):
    """Get rankings for a journal."""
    if name not in DATA["journal_by_name"]:
        return jsonify({"error": "Journal not found"}), 404
    days = request.args.get("days", type=int, default=None)
    top_k = min(request.args.get("top_k", type=int, default=20), 200)
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
_ACRONYM_PATTERNS = [
    (re.compile(r'\b' + re.escape(a.title()) + r'\b'), a)
    for a in sorted(_KNOWN_ACRONYMS, key=len, reverse=True)
]


@app.template_filter("fix_title")
def fix_title_filter(title):
    """Fix ALL CAPS titles to title case, preserving known acronyms."""
    if not title:
        return title
    if title != title.upper():
        return title
    result = title.title()
    for pattern, acronym in _ACRONYM_PATTERNS:
        result = pattern.sub(acronym, result)
    return result


@app.template_filter("top_pct")
def top_pct_filter(percentile):
    """Format percentile as 'Top X%' label."""
    complement = 100 - percentile
    if complement < 0.1:
        return "Top 0.1%"
    elif complement < 1:
        return f"Top {complement:.1f}%"
    else:
        return f"Top {complement:.0f}%"


@app.template_filter("lift_label")
def lift_label_filter(lift):
    """Format lift as '× avg' label."""
    if lift is None or lift < 1.5:
        return ""
    if lift >= 100:
        return f"{lift:.0f}× avg"
    if lift >= 10:
        return f"{lift:.0f}× avg"
    return f"{lift:.1f}× avg"


@app.template_filter("doi_url")
def doi_url_filter(doi, source="medrxiv"):
    """Convert DOI to preprint server URL."""
    if source == "biorxiv":
        return f"https://www.biorxiv.org/content/{doi}"
    return f"https://www.medrxiv.org/content/{doi}"


# ---------- Main ----------

# Load data at import time so gunicorn workers have it ready
_predictions_dir = os.environ.get("PREDICTIONS_DIR", "predictions")
try:
    load_data(_predictions_dir)
except FileNotFoundError as e:
    print(f"FATAL: Could not load predictions data from '{_predictions_dir}': {e}",
          file=sys.stderr)
    sys.exit(1)
print(f"Loaded {DATA['meta']['n_papers']} papers, "
      f"{DATA['meta']['n_journals']} journals")


def main():
    parser = argparse.ArgumentParser(description="medRxiv predictions web app")
    parser.add_argument("--predictions-dir", default=_predictions_dir)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.predictions_dir != _predictions_dir:
        load_data(args.predictions_dir)

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
