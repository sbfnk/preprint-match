#!/usr/bin/env python3
"""How much of a journal match is scope, and how much is format?

Issue #2 asks whether the model matches a preprint to a journal because the
science fits the journal's scope, or because the manuscript's formatting
gives the target away. This script answers it by predicting the journal from
one feature block at a time, on a single shared split, so the blocks can be
read against each other:

  scope      TF-IDF over title + abstract — topic and nothing else
  authors    bag of author names — who wrote it, not what it says
  headings   normalised section headings — the manuscript's house style
  counts     lengths, figure/table/reference counts — no words at all
  citations  bag of cited journal names from the reference list

Headings get their own block rather than being folded into structure,
because they are the most suspicious channel: journals mandate section
structure ("Strengths and limitations of this study" for BMJ Open, "Key
Points" for JAMA), and authors format a manuscript for the target journal
before they post the preprint. That is house style leaking into the input,
not scope. The script also reports which headings are most predictive of
which journal, so the effect can be inspected directly.

Only headings, and the lengths derived from the body, are visible to the
model. The text sent to SPECTER2 is ``title [SEP] abstract [SEP] body_text``,
where ``body_text`` renders the JATS ``<body>`` sections as ``## Heading``
plus paragraphs. Author names never reach it — ``get_full_text_for_embedding``
would add them but is not used by the pipeline — and the reference list lives
in ``<back>``, not ``<body>``. The authors and citations blocks are therefore
controls rather than leaks: citations measures a giveaway the model cannot
currently see, which is the specific one the issue worries about.

Each block goes through the same pipeline the deployed model uses for its
classifier half: reduce to 256 dimensions, then multinomial logistic
regression, scored with acc@1 / acc@10 / MRR.

Usage:
  python3 probe_scope_vs_format.py --metadata finetuned-specter2-v5/embeddings/metadata.json
  python3 probe_scope_vs_format.py --n-train 60000 --n-test 10000
"""

import argparse
import json
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluate_knn import stratified_split_3way
from parse_xml import fix_html_entities, extract_text_from_element

# Headings vary endlessly in wording; collapse them to a small vocabulary so
# "Statistical analysis" and "Statistical analyses" are one feature.
HEADING_STOP = re.compile(r"[^a-z ]+")


def _norm_heading(text):
    return HEADING_STOP.sub(" ", (text or "").lower()).strip()


def extract_features(xml_path):
    """Pull scope, author, structure and citation features from one JATS file.

    Returns None when the file cannot be parsed — a handful of the medRxiv
    XMLs are malformed, and they are not worth special-casing.
    """
    try:
        with open(xml_path, "r", encoding="utf-8") as f:
            content = fix_html_entities(f.read())
        root = ET.fromstring(content)
    except (ET.ParseError, OSError, UnicodeDecodeError):
        return None

    meta = root.find(".//article-meta")
    if meta is None:
        return None

    title_el = meta.find(".//title-group/article-title")
    title = extract_text_from_element(title_el).strip() if title_el is not None else ""
    abs_el = meta.find(".//abstract")
    abstract = extract_text_from_element(abs_el).strip() if abs_el is not None else ""

    authors = []
    for contrib in meta.findall('.//contrib[@contrib-type="author"]'):
        name = contrib.find(".//name")
        if name is None:
            continue
        surname = name.find("surname")
        given = name.find("given-names")
        full = " ".join(x.text for x in (given, surname)
                        if x is not None and x.text)
        if full:
            authors.append(full.lower().replace(" ", "_"))

    # --- structure ---
    body = root.find(".//body")
    headings, n_paras, body_chars = [], 0, 0
    if body is not None:
        for sec in body.findall(".//sec"):
            t = sec.find("title")
            h = _norm_heading(t.text if t is not None else "")
            if h:
                headings.append(h)
        for p in body.findall(".//p"):
            txt = extract_text_from_element(p).strip()
            if txt:
                n_paras += 1
                body_chars += len(txt)

    refs = root.findall(".//ref-list/ref")
    cited, ref_years = [], []
    for ref in refs:
        for src in ref.findall(".//source"):
            if src.text:
                cited.append(_norm_heading(src.text).replace(" ", "_"))
        for yr in ref.findall(".//year"):
            try:
                ref_years.append(int((yr.text or "")[:4]))
            except ValueError:
                pass

    n_words = len(abstract.split())
    numeric = {
        "n_authors": len(authors),
        "n_affiliations": len(meta.findall(".//aff")),
        "n_sections": len(headings),
        "n_paragraphs": n_paras,
        "n_figures": len(root.findall(".//fig")),
        "n_tables": len(root.findall(".//table-wrap")),
        "n_equations": len(root.findall(".//disp-formula")),
        "n_refs": len(refs),
        "abstract_words": n_words,
        "body_chars": body_chars,
        "title_words": len(title.split()),
        "title_has_colon": 1 if ":" in title else 0,
        "mean_para_chars": body_chars / n_paras if n_paras else 0.0,
        # A structured abstract carries its own labelled sections.
        "structured_abstract": 1 if (abs_el is not None
                                     and abs_el.findall(".//sec")) else 0,
        "median_ref_year": float(np.median(ref_years)) if ref_years else 0.0,
        "ref_year_spread": float(np.std(ref_years)) if ref_years else 0.0,
    }

    return {
        "scope": (title + " " + abstract).strip(),
        "authors": " ".join(authors),
        "headings": " ".join(h.replace(" ", "_") for h in headings),
        "citations": " ".join(cited),
        "numeric": numeric,
    }


NUMERIC_KEYS = None  # fixed on first use, so column order stays stable

# Which numeric features the embedded text actually carries. body_text is
# "## Heading" plus paragraphs, so section and paragraph counts and every
# length derived from them are exposed; reference lists, author lists and
# affiliations are not, because they live outside <body>.
LENGTH_COUNTS = ["n_paragraphs", "abstract_words", "body_chars",
                 "title_words", "mean_para_chars"]
VISIBLE_COUNTS = LENGTH_COUNTS + ["n_sections", "title_has_colon",
                                  "structured_abstract"]
HIDDEN_COUNTS = ["n_authors", "n_affiliations", "n_figures", "n_tables",
                 "n_equations", "n_refs", "median_ref_year",
                 "ref_year_spread"]


def _worker(args):
    i, path = args
    return i, extract_features(path)


def build_matrices(feats, train_mask):
    """Vectorise each block, fitting only on training rows."""
    global NUMERIC_KEYS
    NUMERIC_KEYS = sorted(feats[0]["numeric"])

    blocks = {}

    def fit_text(name, key, **kw):
        vec = TfidfVectorizer(**kw)
        vec.fit([f[key] for f, m in zip(feats, train_mask) if m])
        blocks[name] = vec.transform([f[key] for f in feats])

    fit_text("scope", "scope", min_df=5, max_features=200_000,
             ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    fit_text("authors", "authors", min_df=2, token_pattern=r"\S+",
             sublinear_tf=True)
    fit_text("citations", "citations", min_df=3, token_pattern=r"\S+",
             sublinear_tf=True)

    # Headings stay separate from counts so house style can be told apart
    # from sheer document size.
    head_vec = TfidfVectorizer(min_df=5, token_pattern=r"\S+",
                               sublinear_tf=True)
    head_vec.fit([f["headings"] for f, m in zip(feats, train_mask) if m])
    blocks["headings"] = head_vec.transform([f["headings"] for f in feats])

    def numeric(name, keys):
        M = np.array([[f["numeric"][k] for k in keys] for f in feats],
                     dtype=np.float64)
        # Counts are heavy-tailed; log1p before scaling so a 500-reference
        # paper does not dominate the block.
        M = np.log1p(np.clip(M, 0, None))
        scaler = StandardScaler().fit(M[train_mask])
        blocks[name] = sparse.csr_matrix(scaler.transform(M))

    numeric("counts", NUMERIC_KEYS)
    numeric("visible counts", VISIBLE_COUNTS)
    numeric("length", LENGTH_COUNTS)
    numeric("hidden counts", HIDDEN_COUNTS)

    return blocks, head_vec


def discriminative_headings(feats, y, train_mask, vocab, top_journals=15,
                            per_journal=4, min_count=20):
    """Headings that most strongly signal one particular journal.

    Scored by lift: how much more often a heading appears in that journal's
    papers than across the corpus. A heading with high lift and a decent
    absolute count is house style, not subject matter.
    """
    rows = [(f["headings"].split(), j)
            for f, j, m in zip(feats, y, train_mask) if m]
    overall = Counter()
    by_journal = {}
    for heads, journal in rows:
        uniq = set(heads)
        overall.update(uniq)
        by_journal.setdefault(journal, Counter()).update(uniq)

    n_total = len(rows)
    ranked_journals = sorted(by_journal, key=lambda j: -sum(
        1 for _, jj in rows if jj == j))[:top_journals]

    out = []
    for journal in ranked_journals:
        counts = by_journal[journal]
        n_j = sum(1 for _, jj in rows if jj == journal)
        scored = []
        for head, c in counts.items():
            if c < min_count or head not in vocab:
                continue
            lift = (c / n_j) / (overall[head] / n_total)
            scored.append((lift, c / n_j, head))
        scored.sort(reverse=True)
        out.append({
            "journal": journal,
            "n_papers": n_j,
            "headings": [{"heading": h.replace("_", " "),
                          "lift": round(l, 1),
                          "share": round(s, 2)}
                         for l, s, h in scored[:per_journal]],
        })
    return out


def reduce_block(X, train_idx, n_components, seed):
    """Reduce one block to its own dimensions, fitting the SVD on train rows.

    Each block is reduced separately and only then concatenated. Running a
    single SVD over concatenated blocks does not work here: the standardised
    count features have unit variance while TF-IDF entries are tiny, so the
    16-column count block captures almost the whole component budget and the
    200,000-column scope block is discarded. Reducing per block gives each
    an equal share of representation regardless of its native scale.
    """
    n_comp = min(n_components, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    svd.fit(X[train_idx])
    return svd.transform(X), float(svd.explained_variance_ratio_.sum())


def evaluate(Xr_all, y, train_idx, test_idx, label, explained=None):
    """Fit multinomial LR on already-reduced features; score acc@1/@10/MRR."""
    Xr, Xte = Xr_all[train_idx], Xr_all[test_idx]

    # 200 iterations rather than a converged fit: with 1,483 classes a full
    # fit takes ~20 min per block, and the cap is applied identically to
    # every block, so the comparison between them is unaffected. The scope
    # block scores the same at 200 as at 1000, which is the check on this.
    clf = LogisticRegression(C=10.0, max_iter=200)
    clf.fit(Xr, y[train_idx])
    proba = clf.predict_proba(Xte)

    order = np.argsort(proba, axis=1)[:, ::-1]
    ranked = clf.classes_[order]
    truth = y[test_idx][:, None]
    hit = ranked == truth
    rank = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)

    acc1 = float((rank == 1).mean())
    acc10 = float(((rank >= 1) & (rank <= 10)).mean())
    mrr = float(np.where(rank > 0, 1.0 / np.maximum(rank, 1), 0.0).mean())
    extra = "" if explained is None else f"  (explained var {explained:.2f})"
    print(f"  {label:12s} dims={Xr_all.shape[1]:4d}  acc@1={acc1*100:5.1f}%  "
          f"acc@10={acc10*100:5.1f}%  MRR={mrr:.3f}{extra}", file=sys.stderr)
    return {"block": label, "dims": int(Xr_all.shape[1]), "acc1": acc1,
            "acc10": acc10, "mrr": mrr}


def prior_baseline(y, train_idx, test_idx):
    """Rank journals by training frequency, ignoring the paper entirely.

    Without this control the blocks cannot be read: with 1,483 classes of
    very uneven size, always naming the biggest journals already scores
    respectably on acc@10.
    """
    prior = Counter(y[train_idx])
    pos = {j: i + 1 for i, (j, _) in enumerate(prior.most_common())}
    rank = np.array([pos.get(j, 0) for j in y[test_idx]])
    return {"block": "prior only", "dims": 0,
            "acc1": float((rank == 1).mean()),
            "acc10": float(((rank >= 1) & (rank <= 10)).mean()),
            "mrr": float(np.where(rank > 0, 1.0 / np.maximum(rank, 1),
                                  0.0).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="finetuned-specter2-v5/embeddings/metadata.json",
                    help="Embedding metadata giving the model's own dois/journals order")
    ap.add_argument("--xml-index", default="doi_to_xml.json")
    ap.add_argument("--xml-dir", default="xml")
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--min-papers", type=int, default=10)
    ap.add_argument("--components", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="results/probe_features.pkl",
                    help="Where to cache parsed XML features (empty to disable)")
    ap.add_argument("--output", default="results/scope_vs_format.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    md = json.load(open(args.metadata))
    dois, journals = md["dois"], md["journals"]
    index = json.load(open(args.xml_index))
    xml_dir = Path(args.xml_dir)

    # Reproduce the split the deployed model was trained under, so "training
    # papers per journal" means the same thing here as it does there.
    train_idx, _, test_idx = stratified_split_3way(
        journals, val_size=0.1, test_size=0.2, seed=args.seed)
    counts = Counter(journals[i] for i in train_idx)
    eligible = {j for j, c in counts.items() if c >= args.min_papers}
    print(f"{len(eligible)} journals with >={args.min_papers} training papers",
          file=sys.stderr)

    def usable(i):
        return journals[i] in eligible and dois[i] in index

    train_pool = np.array([i for i in train_idx if usable(i)])
    test_pool = np.array([i for i in test_idx if usable(i)])
    print(f"With local XML: {len(train_pool)} train / {len(test_pool)} test",
          file=sys.stderr)

    if len(train_pool) > args.n_train:
        train_pool = rng.choice(train_pool, args.n_train, replace=False)
    if len(test_pool) > args.n_test:
        test_pool = rng.choice(test_pool, args.n_test, replace=False)

    rows = np.concatenate([train_pool, test_pool])
    n_train_rows = len(train_pool)

    # Parsing dominates the runtime, so cache it — the interesting iteration
    # is on feature blocks and models, not on the XML.
    cache = Path(args.cache) if args.cache else None
    feats = None
    if cache and cache.exists():
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        if (blob["seed"] == args.seed and blob["n_train"] == args.n_train
                and blob["n_test"] == args.n_test
                and blob["min_papers"] == args.min_papers):
            feats, rows, n_train_rows = (blob["feats"], blob["rows"],
                                         blob["n_train_rows"])
            print(f"Reusing cached features for {len(feats)} papers "
                  f"({cache})", file=sys.stderr)
        else:
            print("Cache does not match these arguments — reparsing",
                  file=sys.stderr)

    if feats is None:
        paths = [(k, xml_dir / index[dois[i]]) for k, i in enumerate(rows)]
        print(f"Parsing {len(paths)} XML files with {args.workers} workers...",
              file=sys.stderr)
        feats = [None] * len(paths)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done, (k, f) in enumerate(
                    pool.map(_worker, paths, chunksize=64), 1):
                feats[k] = f
                if done % 10000 == 0:
                    print(f"  {done}/{len(paths)}", file=sys.stderr)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump({"feats": feats, "rows": rows,
                             "n_train_rows": n_train_rows, "seed": args.seed,
                             "n_train": args.n_train, "n_test": args.n_test,
                             "min_papers": args.min_papers}, f)
            print(f"Cached parsed features to {cache}", file=sys.stderr)

    ok = np.array([f is not None for f in feats])
    print(f"Parsed {int(ok.sum())}/{len(feats)}", file=sys.stderr)
    feats = [f for f, good in zip(feats, ok) if good]
    rows = rows[ok]
    is_train = np.arange(len(rows)) < ok[:n_train_rows].sum()

    y = np.array([journals[i] for i in rows])
    tr = np.where(is_train)[0]
    te = np.where(~is_train)[0]
    print(f"Final: {len(tr)} train / {len(te)} test, "
          f"{len(set(y[tr]))} journals", file=sys.stderr)

    blocks, head_vec = build_matrices(feats, is_train)
    for name, M in blocks.items():
        print(f"  {name:11s} {M.shape[1]:>7,} features", file=sys.stderr)

    print("\nReducing each block separately...", file=sys.stderr)
    reduced, explained = {}, {}
    for name, M in blocks.items():
        reduced[name], explained[name] = reduce_block(
            M, tr, args.components, args.seed)

    baseline = prior_baseline(y, tr, te)
    print(f"\n  {'prior only':12s} dims=   0  acc@1={baseline['acc1']*100:5.1f}%"
          f"  acc@10={baseline['acc10']*100:5.1f}%  MRR={baseline['mrr']:.3f}",
          file=sys.stderr)

    print("\nSingle blocks:", file=sys.stderr)
    results = [baseline] + [
        evaluate(reduced[n], y, tr, te, n, explained[n])
        for n in ("scope", "authors", "headings", "counts", "visible counts",
                  "hidden counts", "citations")]

    print("\nCombinations (blocks reduced separately, then concatenated):",
          file=sys.stderr)
    combos = {
        "format only": ("headings", "counts"),
        "visible format": ("headings", "visible counts"),
        "scope+format": ("scope", "headings", "counts"),
        # These three separate the two exposed channels: only headings can be
        # removed from the pipeline cleanly, so their split decides whether a
        # rebuild without them is worth the re-embedding cost.
        "scope+visible fmt": ("scope", "headings", "visible counts"),
        "scope+headings": ("scope", "headings"),
        "scope+length": ("scope", "length"),
        "scope+auth": ("scope", "authors"),
        "everything": ("scope", "authors", "headings", "counts", "citations"),
    }
    for label, names in combos.items():
        X = np.hstack([reduced[n] for n in names])
        results.append(evaluate(X, y, tr, te, label))

    print("\nHouse-style headings (lift over corpus rate):", file=sys.stderr)
    train_mask = np.zeros(len(feats), dtype=bool)
    train_mask[tr] = True
    headings = discriminative_headings(feats, y, train_mask,
                                       set(head_vec.vocabulary_))
    for entry in headings:
        top = entry["headings"][:3]
        if not top:
            continue
        shown = "; ".join(f"{h['heading']} (x{h['lift']}, {h['share']:.0%})"
                          for h in top)
        print(f"  {entry['journal'][:34]:36s} {shown}", file=sys.stderr)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "n_train": len(tr), "n_test": len(te),
            "n_journals": len(set(y[tr])),
            "min_papers": args.min_papers,
            "components": args.components,
            "seed": args.seed,
            "results": results,
            "house_style_headings": headings,
        }, f, indent=2)
    print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
