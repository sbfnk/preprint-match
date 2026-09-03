#!/usr/bin/env python3
"""What does the deployed model actually read? Two ablations, same design.

``--variant headings`` deletes every section heading. ``--variant abstract``
drops the body entirely, leaving title and abstract, which is what the daily
refresh feeds papers it has no XML for.

The scope-versus-format probe works on TF-IDF proxies, not on the model that
is served. This measures the real thing: embed the same papers twice with the
fine-tuned SPECTER2 adapter — once with the body as stored, once with the
``## Heading`` lines removed — score both through the saved model, and
compare.

What it answers: how much of the deployed model's output depends on section
headings. If rankings barely move, rebuilding the model without headings
cannot be worth the ~50h of embedding it would cost.

What it does NOT answer: what a model *retrained* without headings would
score. Stripping headings at inference puts the input off-distribution, so
any drop conflates "headings carry signal" with "the input changed shape".
Read the drop as an upper bound on the dependence.

Usage:
  # Locally (no GPU): build a sample from the XML corpus
  python3 probe_heading_sensitivity.py --build-sample \
      --metadata finetuned-specter2-v5/embeddings/metadata.json \
      --output heading_sample.json --n 1500

  # On the cluster (GPU): embed twice, score, report
  python3 probe_heading_sensitivity.py --run \
      --input heading_sample.json --model-dir model-v5 \
      --adapter-path finetuned-specter2-v5/best_adapter \
      --dataset labeled_dataset_v5.json --output results/heading_sensitivity.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# body_text renders each section as "## Title" on its own line.
HEADING_LINE = re.compile(r"^## .*$", re.MULTILINE)


def strip_headings(text):
    """Remove section-heading lines, leaving the paragraphs intact."""
    return re.sub(r"\n{3,}", "\n\n", HEADING_LINE.sub("", text or "")).strip()


def drop_body(text):
    """Discard the body, leaving title and abstract to carry the paper.

    This is what the pipeline feeds any paper it has no XML for, so the
    comparison measures what the full-text backfill is worth to the model
    that is actually served.
    """
    return ""


def build_sample(args):
    """Assemble held-out papers with full text, straight from the XML."""
    from parse_xml import parse_jats_xml
    from evaluate_knn import stratified_split_3way

    md = json.load(open(args.metadata))
    dois, journals, cats = md["dois"], md["journals"], md["categories"]
    index = json.load(open(args.xml_index))
    xml_dir = Path(args.xml_dir)

    train_idx, _, test_idx = stratified_split_3way(
        journals, val_size=0.1, test_size=0.2, seed=args.seed)
    counts = Counter(journals[i] for i in train_idx)
    eligible = {j for j, c in counts.items() if c >= 10}

    pool = [i for i in test_idx
            if journals[i] in eligible and dois[i] in index]
    print(f"{len(pool)} eligible held-out papers with local XML",
          file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(pool)

    records = []
    for i in pool:
        if len(records) >= args.n:
            break
        try:
            parsed = parse_jats_xml(xml_dir / index[dois[i]])
        except Exception:
            continue
        body = parsed.get("body_text")
        # Only papers that actually carry headings can be affected by
        # removing them; the rest would just add noise to the comparison.
        if not body or not HEADING_LINE.search(body):
            continue
        records.append({
            "doi": dois[i],
            "title": parsed.get("title") or "",
            "abstract": parsed.get("abstract") or "",
            "category": cats[i],
            "journal": journals[i],
            "full_text": body,
        })
        if len(records) % 250 == 0:
            print(f"  {len(records)}/{args.n}", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(records, f)
    n_head = np.mean([len(HEADING_LINE.findall(r["full_text"]))
                      for r in records])
    print(f"Wrote {len(records)} papers to {args.output} "
          f"(mean {n_head:.1f} headings each)", file=sys.stderr)


def score(predictor, records, texts_key, adapter_path):
    """Embed one variant of the records and return its probability matrix."""
    from generate_embeddings import load_specter2, select_device
    from generate_embeddings import generate_fulltext_embeddings
    from precompute import compute_proba_matrix

    device = select_device()
    tokenizer, model = load_specter2(device)
    if Path(adapter_path).exists():
        model.load_adapter(str(adapter_path), set_active=True)

    emb = generate_fulltext_embeddings(
        [{"title": r["title"], "abstract": r["abstract"],
          "full_text": r[texts_key]} for r in records],
        tokenizer, model, device, batch_size=32, stride=256)

    cats = [r["category"] for r in records]
    return compute_proba_matrix(emb, cats, predictor, chunk_size=500)


def metrics(proba, truth, classes):
    """acc@1 / acc@10 / MRR against the known journal."""
    order = np.argsort(proba, axis=1)[:, ::-1]
    ranked = classes[order]
    hit = ranked == truth[:, None]
    rank = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)
    return {
        "acc1": float((rank == 1).mean()),
        "acc10": float(((rank >= 1) & (rank <= 10)).mean()),
        "mrr": float(np.where(rank > 0, 1.0 / np.maximum(rank, 1), 0.0).mean()),
        "rank": rank,
        "top1": ranked[:, 0],
        "top5": ranked[:, :5],
    }


def run(args):
    from predict_journal import JournalPredictor

    records = json.load(open(args.input))
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} papers", file=sys.stderr)

    variant = getattr(args, "variant", "headings")
    transform = {"headings": strip_headings, "abstract": drop_body}[variant]
    for r in records:
        r["stripped"] = transform(r["full_text"])
    label = {"headings": "headings stripped",
             "abstract": "title + abstract only"}[variant]

    predictor = JournalPredictor.load(args.model_dir, args.dataset)
    classes = np.array(predictor.restricted_classes)
    truth = np.array([r["journal"] for r in records])

    print("\n=== with headings ===", file=sys.stderr)
    p_with = score(predictor, records, "full_text", args.adapter_path)
    print(f"\n=== {label} ===", file=sys.stderr)
    p_without = score(predictor, records, "stripped", args.adapter_path)

    m_with = metrics(p_with, truth, classes)
    m_without = metrics(p_without, truth, classes)

    top1_same = float((m_with["top1"] == m_without["top1"]).mean())
    top5_overlap = float(np.mean([
        len(set(a) & set(b)) / 5
        for a, b in zip(m_with["top5"], m_without["top5"])]))
    # Rank movement of the true journal; 0 means "not in the list at all".
    both = (m_with["rank"] > 0) & (m_without["rank"] > 0)
    rank_shift = float(np.mean(np.abs(
        m_with["rank"][both] - m_without["rank"][both])))
    cos = float(np.mean(np.sum(p_with * p_without, axis=1) / (
        np.linalg.norm(p_with, axis=1) * np.linalg.norm(p_without, axis=1))))

    out = {
        "variant": variant,
        "n_papers": len(records),
        "with_headings": {k: m_with[k] for k in ("acc1", "acc10", "mrr")},
        "without_headings": {k: m_without[k] for k in ("acc1", "acc10", "mrr")},
        "top1_unchanged": top1_same,
        "top5_overlap": top5_overlap,
        "mean_abs_rank_shift_of_true_journal": rank_shift,
        "mean_cosine_between_prob_vectors": cos,
    }

    print("\n" + "=" * 62, file=sys.stderr)
    for lbl, m in (("full text", m_with), (label, m_without)):
        print(f"  {lbl:22s} acc@1={m['acc1']*100:5.1f}%  "
              f"acc@10={m['acc10']*100:5.1f}%  MRR={m['mrr']:.3f}",
              file=sys.stderr)
    d1 = (m_without["acc1"] - m_with["acc1"]) * 100
    d10 = (m_without["acc10"] - m_with["acc10"]) * 100
    print(f"  {'delta':16s} acc@1={d1:+5.1f}pp  acc@10={d10:+5.1f}pp",
          file=sys.stderr)
    print(f"\n  top-1 unchanged: {top1_same:.1%}   "
          f"top-5 overlap: {top5_overlap:.1%}", file=sys.stderr)
    print(f"  mean |rank shift| of true journal: {rank_shift:.2f}",
          file=sys.stderr)
    print(f"  mean cosine between probability vectors: {cos:.4f}",
          file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("headings", "abstract"),
                    default="headings",
                    help="what to remove: section headings, or the whole body")
    ap.add_argument("--build-sample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--metadata",
                    default="finetuned-specter2-v5/embeddings/metadata.json")
    ap.add_argument("--xml-index", default="doi_to_xml.json")
    ap.add_argument("--xml-dir", default="xml")
    ap.add_argument("--input", default="heading_sample.json")
    ap.add_argument("--model-dir", default="model-v5")
    ap.add_argument("--adapter-path",
                    default="finetuned-specter2-v5/best_adapter")
    ap.add_argument("--dataset", default="labeled_dataset.json")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/heading_sensitivity.json")
    args = ap.parse_args()

    if args.build_sample:
        build_sample(args)
    elif args.run:
        run(args)
    else:
        ap.error("pass --build-sample or --run")


if __name__ == "__main__":
    main()
