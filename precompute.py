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
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from extract_labeled_data import fetch_preprints


def _authors_to_str(authors):
    """Normalise an authors field (string or list of dicts) to a string."""
    if isinstance(authors, list):
        return " ".join(
            f"{a.get('given_names', '')} {a.get('surname', '')}"
            for a in authors if isinstance(a, dict))
    return authors or ""


# Percentile levels the grid stores, in hundredths of a point across the
# whole range. The UI rounds to whole points below the 99th and to tenths
# above it, so this keeps the rendered label identical to what an exact rank
# would produce; a coarser grid disagreed on about a quarter of labels.
PERCENTILE_LEVELS = np.round(np.arange(0.0, 100.0 + 1e-9, 0.01), 2)


def build_percentile_grid(output_dir, proba):
    """Store per-journal quantile thresholds for percentile ranking.

    The web app used to keep ``np.sort(proba, axis=0)`` in memory purely to
    binary-search one column per lookup, which cost as much RAM as the
    probability matrix itself (~600MB). The thresholds reproduce the same
    ranking to within the grid resolution for about a thousandth of that.

    Columns are quantiled one at a time so peak memory stays at one column.
    """
    output_dir = Path(output_dir)
    n_journals = proba.shape[1]
    # float16 to match the matrix being searched, so comparisons are exact in
    # the same precision. Casting keeps the thresholds non-decreasing.
    grid = np.empty((n_journals, len(PERCENTILE_LEVELS)), dtype=np.float16)
    qs = PERCENTILE_LEVELS / 100.0
    for j in range(n_journals):
        grid[j] = np.quantile(proba[:, j].astype(np.float32), qs)
    np.savez_compressed(output_dir / "percentile_grid.npz",
                        levels=PERCENTILE_LEVELS.astype(np.float32),
                        thresholds=grid)
    print(f"  percentile_grid.npz {grid.shape} "
          f"({grid.nbytes / 1e6:.1f}MB in memory)", file=sys.stderr)


def build_web_artifacts(output_dir):
    """Build the slimmed-down artifacts the web app serves.

    Reads the canonical ``papers.json`` + ``proba_matrix.npz`` and writes:
      * ``proba_matrix.npz`` rewritten as float16 (halves the matrix in RAM)
      * ``percentile_grid.npz`` — per-journal quantile thresholds, so the web
        app can rank a probability without holding a sorted copy of the matrix
      * ``abstracts.db`` — SQLite FTS5 index over title/abstract/authors,
        keyed by DOI, so abstracts live on disk instead of the Python heap
      * ``papers_slim.json`` — papers list with the ``abstract`` field dropped

    Safe to run standalone on an existing predictions dir (no GPU/fetch).
    """
    output_dir = Path(output_dir)

    # --- float16 probability matrix ---
    proba_path = output_dir / "proba_matrix.npz"
    if proba_path.exists():
        proba = np.load(proba_path)["proba"]
        if proba.dtype != np.float16:
            np.savez_compressed(proba_path, proba=proba.astype(np.float16))
            print(f"  Rewrote {proba_path.name} as float16 "
                  f"({proba.shape})", file=sys.stderr)
        else:
            print(f"  {proba_path.name} already float16", file=sys.stderr)
        build_percentile_grid(output_dir, proba)

    # --- neighbour evidence placeholders ---
    # The Dockerfile copies these unconditionally, so they must exist even
    # when the model is too old to produce them. An empty artifact fails the
    # web app's row-count check and is ignored at load.
    if not (output_dir / "neighbours.npz").exists():
        save_neighbours(output_dir,
                        (np.empty((0, NB_JOURNALS), dtype=np.int16),
                         np.empty((0, NB_JOURNALS, NB_PER_JOURNAL), dtype=np.int32),
                         np.empty((0, NB_JOURNALS, NB_PER_JOURNAL), dtype=np.float16)),
                        [])

    # --- abstracts.db + papers_slim.json ---
    papers_path = output_dir / "papers.json"
    if not papers_path.exists():
        print("  No papers.json — skipping abstracts.db/papers_slim.json",
              file=sys.stderr)
        return
    with open(papers_path) as f:
        papers = json.load(f)

    db_path = output_dir / "abstracts.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    # FTS5 table for keyword search (doi UNINDEXED is fine here — we read doi
    # from MATCH results, never filter by it).
    conn.execute(
        "CREATE VIRTUAL TABLE papers_fts USING fts5("
        "doi UNINDEXED, title, abstract, authors)")
    # Separate table with doi PRIMARY KEY for fast per-DOI display lookups.
    # Without this, `WHERE doi IN (...)` against the FTS table is a full scan,
    # which is unusably slow on a large DB over network/slow disk.
    conn.execute("CREATE TABLE abstracts (doi TEXT PRIMARY KEY, abstract TEXT)")
    rows = [(p.get("doi", ""), p.get("title", ""), p.get("abstract", ""),
             _authors_to_str(p.get("authors", "")))
            for p in papers]
    conn.executemany(
        "INSERT INTO papers_fts (doi, title, abstract, authors) "
        "VALUES (?, ?, ?, ?)", rows)
    conn.executemany(
        "INSERT OR IGNORE INTO abstracts (doi, abstract) VALUES (?, ?)",
        [(r[0], r[2]) for r in rows])
    conn.commit()
    conn.close()
    print(f"  Wrote {db_path.name} ({len(papers)} papers, FTS5 + doi index)",
          file=sys.stderr)

    slim_path = output_dir / "papers_slim.json"
    # Drop fields the webapp never serves from papers_slim: abstract (served
    # from abstracts.db), and full_text/xml_file (only used at embedding time).
    _slim_drop = {"abstract", "full_text", "xml_file"}
    slim = [{k: v for k, v in p.items() if k not in _slim_drop} for p in papers]
    with open(slim_path, "w") as f:
        json.dump(slim, f)
    print(f"  Wrote {slim_path.name} ({len(slim)} papers, no abstracts)",
          file=sys.stderr)

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


def embed_papers(papers, adapter_path="finetuned-specter2/best_adapter",
                 checkpoint_dir=None):
    """Embed papers using fine-tuned SPECTER2.

    If checkpoint_dir is given, embedding is checkpointed every 1000 records
    and resumes from any existing checkpoint — essential for large prediction
    sets (~200k papers, many hours) that may outlive a single job's walltime.
    """
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

    # Track which papers were embedded from full text. Preprints fetched from
    # the API carry title and abstract only, so an embedding built from them
    # sits on a different distribution from the full-text ones the model was
    # trained on. Without recording it at this point the distinction is
    # unrecoverable later: papers.json does not keep full_text.
    used_fulltext = np.array([bool(r["full_text"]) for r in records], dtype=bool)

    start_idx = 0
    existing = None
    if checkpoint_dir is not None:
        from generate_embeddings import _load_checkpoint
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ckpt = _load_checkpoint(Path(checkpoint_dir))
        if ckpt is not None:
            existing, start_idx = ckpt
            print(f"Resuming embedding from record {start_idx}", file=sys.stderr)

    embeddings = generate_fulltext_embeddings(
        records, tokenizer, model, device, batch_size=32, stride=256,
        checkpoint_dir=checkpoint_dir, checkpoint_every=1000,
        start_idx=start_idx, existing_embeddings=existing)
    return embeddings, used_fulltext


# Neighbour evidence: how many top journals get evidence, and how many
# example papers each. Both feed fixed-shape arrays, so keep them small.
NB_JOURNALS = 5
NB_PER_JOURNAL = 3


def _train_rows_by_journal(predictor):
    """Map eligible-journal column index → training rows published there."""
    col = {j: i for i, j in enumerate(predictor.restricted_classes)}
    groups = {}
    for row, journal in enumerate(predictor.train_journals):
        k = col.get(journal)
        if k is not None:
            groups.setdefault(k, []).append(row)
    return {k: np.array(v) for k, v in groups.items()}


def _chunk_neighbours(sim, proba_chunk, groups):
    """For each paper, the most similar training papers in its top journals.

    Reuses the similarity matrix the kNN step has already computed, so this
    costs a partial sort per (paper, journal) rather than another matmul.
    """
    n = proba_chunk.shape[0]
    j_out = np.full((n, NB_JOURNALS), -1, dtype=np.int16)
    r_out = np.full((n, NB_JOURNALS, NB_PER_JOURNAL), -1, dtype=np.int32)
    s_out = np.zeros((n, NB_JOURNALS, NB_PER_JOURNAL), dtype=np.float16)

    for i in range(n):
        top_j = np.argpartition(proba_chunk[i], -NB_JOURNALS)[-NB_JOURNALS:]
        top_j = top_j[np.argsort(proba_chunk[i][top_j])[::-1]]
        j_out[i] = top_j
        for slot, k in enumerate(top_j):
            rows = groups.get(int(k))
            if rows is None or not len(rows):
                continue
            m = min(NB_PER_JOURNAL, len(rows))
            best = rows[np.argpartition(sim[i, rows], -m)[-m:]]
            best = best[np.argsort(sim[i, best])[::-1]]
            r_out[i, slot, :m] = best
            s_out[i, slot, :m] = sim[i, best]

    return j_out, r_out, s_out


def compute_proba_matrix(emb, categories, predictor, chunk_size=2000,
                         with_neighbours=False):
    """Compute full probability matrix: n_papers × n_eligible_journals.

    Processes in chunks to limit memory usage — the full similarity matrix
    (n_papers × n_train) can be 10s of GB and doesn't fit in CI runners.

    With ``with_neighbours``, also returns the nearest training papers per
    top journal, for the "similar papers" evidence on paper pages.
    """
    from evaluate_knn import predict_knn
    from train_classifier import build_feature_matrix
    from calibrate import ensemble_proba_matrix
    from predict_journal import restrict_and_renormalize, temperature_scale

    n = emb.shape[0]
    n_eligible = int(predictor.eligible_mask.sum())
    proba_all = np.empty((n, n_eligible), dtype=np.float32)

    groups = _train_rows_by_journal(predictor) if with_neighbours else None
    nb_j = nb_r = nb_s = None
    if with_neighbours:
        # journal_idx is int16 to keep the artifact small; that caps the
        # eligible-journal count well above anything the model produces.
        assert n_eligible <= np.iinfo(np.int16).max, (
            f"{n_eligible} eligible journals exceeds the int16 index")
        nb_j = np.empty((n, NB_JOURNALS), dtype=np.int16)
        nb_r = np.empty((n, NB_JOURNALS, NB_PER_JOURNAL), dtype=np.int32)
        nb_s = np.empty((n, NB_JOURNALS, NB_PER_JOURNAL), dtype=np.float16)

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

        if with_neighbours:
            j, r, s = _chunk_neighbours(sim, proba_chunk, groups)
            nb_j[start:end], nb_r[start:end], nb_s[start:end] = j, r, s

        print(f"  Scored {end}/{n} papers", file=sys.stderr)

    if with_neighbours:
        return proba_all, (nb_j, nb_r, nb_s)
    return proba_all


def save_neighbours(output_dir, neighbours, train_dois):
    """Write the neighbour evidence arrays plus their DOI side table.

    Rows are stored as indices into ``train_dois`` rather than DOI strings:
    the same training papers recur across many preprints, so indices cut the
    artifact from hundreds of MB to tens.
    """
    nb_j, nb_r, nb_s = neighbours
    output_dir = Path(output_dir)
    np.savez_compressed(output_dir / "neighbours.npz",
                        journal_idx=nb_j, train_idx=nb_r, sim=nb_s)
    with open(output_dir / "neighbour_dois.json", "w") as f:
        json.dump(list(train_dois), f)
    print(f"  neighbours.npz {nb_r.shape} + {len(train_dois)} training DOIs",
          file=sys.stderr)


def _guard_full_reembed(papers, output_dir, args):
    """Refuse a re-embed that would silently drop full-text coverage.

    Full text reaches the pipeline only through a bulk XML backfill; papers
    fetched from the API carry title and abstract only, and papers.json does
    not keep the text. So deleting embeddings.npz and rebuilding from the
    current papers.json converts the whole corpus to abstract-only, which is
    worth about -0.7pp acc@1 and -2pp acc@10 (RESULTS.md) and is invisible
    once done. Fail loudly instead, unless the caller says they mean it.
    """
    prev = output_dir / "embeddings.npz"
    if prev.exists():
        return
    with_text = sum(1 for p in papers if p.get("full_text"))
    if with_text or getattr(args, "allow_abstract_only", False):
        return
    print(
        "\nREFUSING to embed: no paper in papers.json carries full_text, so\n"
        "this would rebuild the whole corpus from title+abstract only and\n"
        "quietly discard the full-text signal.\n\n"
        "Run the XML backfill first (see pipeline/README.md), or pass\n"
        "--allow-abstract-only if abstract-only embeddings are intended.\n",
        file=sys.stderr)
    raise SystemExit(2)


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
    parser.add_argument("--init-fulltext-flags", action="store_true",
                        help="Record which existing embeddings used full text, "
                             "taken from papers.json, without re-embedding")
    parser.add_argument("--embed-only", action="store_true",
                        help="Stop after writing embeddings; leave scoring to "
                             "the refresh")
    parser.add_argument("--fulltext-index",
                        help="With --init-fulltext-flags, derive provenance "
                             "from this doi->xml index (snapshot it before a "
                             "backfill extends it) instead of papers.json")
    parser.add_argument("--reembed-fulltext-gained", action="store_true",
                        help="Re-embed only papers that have gained full text "
                             "since they were last embedded")
    parser.add_argument("--allow-abstract-only", action="store_true",
                        help="Permit a full re-embed with no full text "
                             "available (see _guard_full_reembed)")
    parser.add_argument("--no-neighbours", action="store_true",
                        help="Skip the 'similar papers' evidence artifact")
    parser.add_argument("--web-artifacts-only", action="store_true",
                        help="Only (re)build web artifacts (float16 matrix, "
                             "abstracts.db, papers_slim.json) from existing "
                             "predictions dir; no fetch/embed/GPU")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.web_artifacts_only:
        print("Building web artifacts from existing predictions...",
              file=sys.stderr)
        build_web_artifacts(output_dir)
        return

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
    ft_flags = None
    if emb_path.exists() and not args.fetch_only:
        _e = np.load(emb_path)
        emb = _e["embeddings"]
        # Older embedding files predate provenance tracking; treat them as
        # unknown rather than asserting they were abstract-only.
        ft_flags = _e["used_fulltext"] if "used_fulltext" in _e else None

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

    # Adopt the current papers.json full-text state as the provenance record
    # for an embedding file written before provenance was tracked. Only sound
    # when papers.json reflects the XML the embeddings were actually built
    # from, so it is a deliberate one-off rather than automatic.
    if getattr(args, "init_fulltext_flags", False):
        if emb is None:
            print("No embeddings.npz to annotate.", file=sys.stderr)
            raise SystemExit(2)
        # papers.json does not retain full_text, so by default there would be
        # nothing to read. Deriving the flags from the DOI->XML index used at
        # the time is equivalent and far cheaper than reparsing every file:
        # the build attaches full text to exactly those papers the index
        # resolves. Snapshot that index before a backfill extends it.
        index_path = getattr(args, "fulltext_index", None)
        if index_path:
            with open(index_path) as f:
                indexed = set(json.load(f))
            flags = np.array([p.get("doi") in indexed for p in papers[:len(emb)]],
                             dtype=bool)
            print(f"Deriving provenance from {index_path} "
                  f"({len(indexed):,} DOIs indexed)", file=sys.stderr)
        else:
            flags = np.array([bool(p.get("full_text")) for p in papers[:len(emb)]],
                             dtype=bool)
        if len(flags) != len(emb):
            print(f"papers.json ({len(papers)}) shorter than embeddings "
                  f"({len(emb)}) — refusing to guess.", file=sys.stderr)
            raise SystemExit(2)
        np.savez_compressed(emb_path, embeddings=emb, used_fulltext=flags)
        print(f"Recorded full-text provenance for {len(flags)} embeddings "
              f"({int(flags.sum())} with full text, "
              f"{flags.sum()/len(flags)*100:.1f}%)", file=sys.stderr)
        return

    # ---------- Embed ----------
    # Find papers that need embedding
    if emb is not None and emb.shape[0] < len(papers):
        n_existing = emb.shape[0]
        papers_to_embed = papers[n_existing:]
        print(f"Embedding {len(papers_to_embed)} new papers "
              f"({n_existing} already embedded)...", file=sys.stderr)
        new_emb, new_ft = embed_papers(
            papers_to_embed, args.adapter_path,
            checkpoint_dir=output_dir / "emb_checkpoint")
        emb = np.concatenate([emb, new_emb], axis=0)
        if ft_flags is not None and len(ft_flags) == n_existing:
            ft_flags = np.concatenate([ft_flags, new_ft])
        else:
            # No record of how the existing rows were built. Inventing False
            # would assert they are abstract-only, which is usually wrong and
            # overwrites the real answer once it is written back. Leave the
            # provenance unknown instead; --init-fulltext-flags restores it.
            ft_flags = None
            print("No used_fulltext array for the existing embeddings — "
                  "leaving provenance unrecorded rather than guessing. "
                  "Run --init-fulltext-flags to restore it.", file=sys.stderr)
    elif emb is None:
        _guard_full_reembed(papers, output_dir, args)
        print(f"Embedding all {len(papers)} papers...", file=sys.stderr)
        emb, ft_flags = embed_papers(
            papers, args.adapter_path,
            checkpoint_dir=output_dir / "emb_checkpoint")
    else:
        print(f"All {len(papers)} papers already embedded.", file=sys.stderr)

    # ---------- Backfill: re-embed papers that have gained full text ----------
    # Embeddings are otherwise written once and never revisited, so a paper
    # first embedded from title+abstract keeps that embedding even after an
    # XML backfill gives it a body. Re-embedding only the affected rows costs
    # hours rather than the ~30 a full pass would.
    if emb is not None and getattr(args, "reembed_fulltext_gained", False):
        if ft_flags is None or len(ft_flags) != len(emb):
            print("Cannot target a backfill: embeddings.npz has no "
                  "used_fulltext array. Run --init-fulltext-flags first "
                  "(see pipeline/README.md).", file=sys.stderr)
            raise SystemExit(2)
        gained = [i for i, p in enumerate(papers[:len(emb)])
                  if p.get("full_text") and not ft_flags[i]]
        if not gained:
            print("No papers have gained full text since they were embedded.",
                  file=sys.stderr)
        else:
            print(f"Re-embedding {len(gained)} papers that gained full text...",
                  file=sys.stderr)
            re_emb, re_ft = embed_papers(
                [papers[i] for i in gained], args.adapter_path,
                checkpoint_dir=output_dir / "reembed_checkpoint")
            emb[gained] = re_emb
            ft_flags[gained] = re_ft
            print(f"  spliced {int(re_ft.sum())} full-text embeddings into "
                  f"place", file=sys.stderr)

    if ft_flags is not None and len(ft_flags) == len(emb):
        n_ft = int(ft_flags.sum())
        print(f"Full-text coverage: {n_ft}/{len(ft_flags)} embeddings "
              f"({n_ft / len(ft_flags) * 100:.1f}%)", file=sys.stderr)

    if emb is None or len(papers) == 0:
        print("No papers to score.", file=sys.stderr)
        return

    if getattr(args, "embed_only", False):
        # Used for a GPU backfill: the embeddings are the expensive part and
        # the only thing that needs this machine. Scoring happens in the
        # regular refresh, which would redo it anyway.
        if ft_flags is not None and len(ft_flags) == len(emb):
            np.savez_compressed(emb_path, embeddings=emb, used_fulltext=ft_flags)
        else:
            np.savez_compressed(emb_path, embeddings=emb)
        print(f"Embed-only: wrote {emb_path} ({emb.shape}); skipping scoring.",
              file=sys.stderr)
        return

    # Save embeddings, with the full-text provenance alongside them
    if ft_flags is not None and len(ft_flags) == len(emb):
        np.savez_compressed(emb_path, embeddings=emb, used_fulltext=ft_flags)
    else:
        np.savez_compressed(emb_path, embeddings=emb)

    # ---------- Score ----------
    from predict_journal import JournalPredictor
    predictor = JournalPredictor.load(args.model_dir, args.dataset)

    print(f"Computing {len(papers)} × {len(predictor.restricted_classes)} "
          f"probability matrix...", file=sys.stderr)
    categories = [p.get("category", "") for p in papers]

    # Neighbour evidence needs train_dois, which models saved before this
    # feature lack; those still score normally, just without evidence.
    want_neighbours = not args.no_neighbours and bool(
        getattr(predictor, "train_dois", None))
    if not args.no_neighbours and not want_neighbours:
        print("  Model has no train_dois — skipping neighbour evidence.",
              file=sys.stderr)

    result = compute_proba_matrix(emb, categories, predictor,
                                  with_neighbours=want_neighbours)
    proba, neighbours = result if want_neighbours else (result, None)

    # ---------- Save ----------
    # Full probability matrix
    np.savez_compressed(output_dir / "proba_matrix.npz", proba=proba)
    if neighbours is not None:
        save_neighbours(output_dir, neighbours, predictor.train_dois)

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

    # Web artifacts: float16 matrix, abstracts.db, papers_slim.json
    print("Building web artifacts...", file=sys.stderr)
    build_web_artifacts(output_dir)

    print(f"\nPrecomputed:", file=sys.stderr)
    print(f"  Papers: {len(papers)}", file=sys.stderr)
    print(f"  Journals: {len(journals)}", file=sys.stderr)
    if dates:
        print(f"  Date range: {dates[0]} to {dates[-1]}", file=sys.stderr)
    print(f"  Matrix: {proba.shape}", file=sys.stderr)
    print(f"  Output: {output_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
