#!/usr/bin/env python3
"""Does adding cited journals improve the real classifier? (issue #14)

Experiment 5 in RESULTS.md measured cited journals through a TF-IDF proxy.
Experiment 5b then showed the proxy misleads: it credited section headings
with +1.6pp acc@1 where the deployed model turned out to lose only 0.5pp
without them. So this measures the citation block against the actual
classifier the site serves — same embeddings, same split, same
hyperparameters — and changes exactly one thing.

Baseline reproduces the deployed classifier: PCA-256 over the fine-tuned
SPECTER2 embeddings plus a category one-hot, multinomial logistic
regression at C=10. The treatment appends an SVD-reduced bag of cited
journal names. Papers whose XML is missing (~7%) get a zero citation block,
which is what inference would see for them anyway.

Usage:
  python3 evaluate_citations.py --metadata finetuned-specter2-v5/embeddings/metadata.json \
      --embeddings finetuned-specter2-v5/embeddings/embeddings.npz \
      --citations citations.jsonl.gz
"""

import argparse
import gzip
import json
import sys
import time
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluate_knn import stratified_split_3way
from train_classifier import build_feature_matrix


def metrics(proba, classes, truth, label, dims):
    order = np.argsort(proba, axis=1)[:, ::-1]
    ranked = classes[order]
    hit = ranked == truth[:, None]
    rank = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)
    out = {
        "block": label, "dims": int(dims),
        "acc1": float((rank == 1).mean()),
        "acc5": float(((rank >= 1) & (rank <= 5)).mean()),
        "acc10": float(((rank >= 1) & (rank <= 10)).mean()),
        "mrr": float(np.where(rank > 0, 1.0 / np.maximum(rank, 1), 0.0).mean()),
    }
    print(f"  {label:22} dims={out['dims']:4d}  acc@1={out['acc1']*100:5.1f}%  "
          f"acc@5={out['acc5']*100:5.1f}%  acc@10={out['acc10']*100:5.1f}%  "
          f"MRR={out['mrr']:.3f}", file=sys.stderr)
    return out


def fit_and_score(X, y, tr, te, classes_ref, label, C, max_iter):
    t0 = time.time()
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])
    r = metrics(proba, clf.classes_, y[te], label, X.shape[1])
    print(f"    ({time.time() - t0:.0f}s)", file=sys.stderr)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--citations", default="citations.jsonl.gz")
    ap.add_argument("--min-papers", type=int, default=10)
    ap.add_argument("--pca-components", type=int, default=256)
    ap.add_argument("--cite-components", type=int, default=256)
    ap.add_argument("--classifier-C", type=float, default=10.0)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/citation_ablation.json")
    args = ap.parse_args()

    md = json.load(open(args.metadata))
    dois, journals, cats = md["dois"], md["journals"], md["categories"]
    emb = np.load(args.embeddings)["embeddings"]
    assert emb.shape[0] == len(dois), (emb.shape, len(dois))
    print(f"{emb.shape[0]} papers, {emb.shape[1]}-dim embeddings", file=sys.stderr)

    train_idx, _, test_idx = stratified_split_3way(
        journals, val_size=0.1, test_size=0.2, seed=args.seed)
    counts = Counter(journals[i] for i in train_idx)
    eligible = {j for j, c in counts.items() if c >= args.min_papers}
    tr = np.array([i for i in train_idx if journals[i] in eligible])
    te = np.array([i for i in test_idx if journals[i] in eligible])
    print(f"{len(eligible)} eligible journals | {len(tr)} train / {len(te)} test",
          file=sys.stderr)

    y = np.array(journals)
    cat_to_idx = {c: i + 1 for i, c in enumerate(sorted(set(cats)))}

    # --- baseline: what the site serves ---
    print("\nReducing embeddings...", file=sys.stderr)
    pca = PCA(n_components=args.pca_components, random_state=args.seed)
    pca.fit(emb[tr])
    emb_r = pca.transform(emb)
    X_base = build_feature_matrix(emb_r, cats, cat_to_idx, True)
    print(f"  baseline features: {X_base.shape[1]} "
          f"(explained var {pca.explained_variance_ratio_.sum():.2f})",
          file=sys.stderr)

    # --- citation block ---
    print("Building citation block...", file=sys.stderr)
    cited = {}
    with gzip.open(args.citations, "rt") as f:
        for line in f:
            r = json.loads(line)
            cited[r["doi"]] = " ".join(x.replace(" ", "_") for x in r["cited"])
    docs = [cited.get(d, "") for d in dois]
    have = sum(1 for x in docs if x)
    print(f"  {have}/{len(docs)} papers with citations "
          f"({have/len(docs)*100:.1f}%)", file=sys.stderr)

    vec = TfidfVectorizer(min_df=3, token_pattern=r"\S+", sublinear_tf=True)
    vec.fit([docs[i] for i in tr])
    Xc = vec.transform(docs)
    svd = TruncatedSVD(n_components=args.cite_components, random_state=args.seed)
    svd.fit(Xc[tr])
    cite_r = svd.transform(Xc)
    print(f"  {Xc.shape[1]} distinct cited journals -> {cite_r.shape[1]} dims "
          f"(explained var {svd.explained_variance_ratio_.sum():.2f})",
          file=sys.stderr)

    X_cite = np.hstack([X_base, cite_r.astype(np.float32)])

    print(f"\nFitting (C={args.classifier_C}, max_iter={args.max_iter}):",
          file=sys.stderr)
    results = [
        fit_and_score(X_base, y, tr, te, None, "baseline (deployed)",
                      args.classifier_C, args.max_iter),
        fit_and_score(X_cite, y, tr, te, None, "+ cited journals",
                      args.classifier_C, args.max_iter),
    ]
    b, c = results
    print(f"\n  marginal: acc@1 {(c['acc1']-b['acc1'])*100:+.1f}pp  "
          f"acc@10 {(c['acc10']-b['acc10'])*100:+.1f}pp  "
          f"MRR {c['mrr']-b['mrr']:+.3f}", file=sys.stderr)

    from pathlib import Path
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_train": len(tr), "n_test": len(te),
               "n_journals": len(eligible), "citation_coverage": have / len(docs),
               "results": results}, open(out, "w"), indent=2)
    print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
