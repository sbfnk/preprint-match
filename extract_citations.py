#!/usr/bin/env python3
"""Extract the journals each preprint cites, from the local JATS XML corpus.

Reference lists live in JATS ``<back>`` and never reach the embedding model,
which reads ``<body>`` only. Measured against the model's other inputs they
carry as much journal signal as the whole title and abstract, and add
+5.2pp acc@10 on top of everything it can currently see (RESULTS.md,
Experiment 5) — so they are the largest unused feature available. See #14.

Writes one gzipped JSON record per paper so the result can be reused across
experiments without reparsing 170k XML files.

Usage:
  python3 extract_citations.py --dois finetuned-specter2-v5/embeddings/metadata.json \
      --output citations.jsonl.gz
  python3 extract_citations.py --dois metadata.json --limit 2000   # quick trial
"""

import argparse
import gzip
import json
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from parse_xml import fix_html_entities

# Journal names arrive with wildly inconsistent punctuation, casing and
# abbreviation. Collapsing to lowercase alphanumerics merges the easy cases
# ("PLoS ONE" / "PLoS One" / "PLOS ONE.") without pretending to resolve
# abbreviations, which needs a real title list.
_STRIP = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")


def normalise(name):
    return _SPACE.sub(" ", _STRIP.sub(" ", (name or "").lower())).strip()


def cited_journals(xml_path):
    """Return (normalised cited journal names, publication years) for one file."""
    try:
        with open(xml_path, "r", encoding="utf-8") as f:
            root = ET.fromstring(fix_html_entities(f.read()))
    except (ET.ParseError, OSError, UnicodeDecodeError):
        return None

    names, years = [], []
    for ref in root.findall(".//ref-list/ref"):
        for src in ref.findall(".//source"):
            n = normalise("".join(src.itertext()))
            if n:
                names.append(n)
        for yr in ref.findall(".//year"):
            txt = "".join(yr.itertext()).strip()[:4]
            if txt.isdigit():
                years.append(int(txt))
    return names, years


def _worker(args):
    doi, path = args
    got = cited_journals(path)
    if got is None:
        return doi, None
    names, years = got
    return doi, {"doi": doi, "cited": names, "years": years}


def load_dois(path):
    """Accept the embedding metadata, a labelled dataset, or a plain DOI list."""
    with open(path) as f:
        blob = json.load(f)
    if isinstance(blob, dict) and "dois" in blob:
        return blob["dois"]
    if isinstance(blob, list):
        if blob and isinstance(blob[0], dict):
            return [r.get("preprint_doi") or r.get("doi") for r in blob]
        return blob
    raise SystemExit(f"Cannot find DOIs in {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dois", required=True,
                    help="embedding metadata.json, labelled dataset, or DOI list")
    ap.add_argument("--xml-index", default="doi_to_xml.json")
    ap.add_argument("--xml-dir", default="xml")
    ap.add_argument("--output", default="citations.jsonl.gz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dois = load_dois(args.dois)
    index = json.load(open(args.xml_index))
    xml_dir = Path(args.xml_dir)

    jobs = [(d, xml_dir / index[d]) for d in dois if d in index]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(dois)} DOIs, {len(jobs)} with local XML "
          f"({len(jobs) / max(len(dois), 1) * 100:.1f}%)", file=sys.stderr)

    n_ok = n_fail = n_empty = 0
    with gzip.open(args.output, "wt") as out, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (doi, rec) in enumerate(
                pool.map(_worker, jobs, chunksize=64), 1):
            if rec is None:
                n_fail += 1
            elif not rec["cited"]:
                n_empty += 1
            else:
                out.write(json.dumps(rec) + "\n")
                n_ok += 1
            if i % 20000 == 0:
                print(f"  {i}/{len(jobs)}", file=sys.stderr)

    print(f"\nwrote {n_ok} records to {args.output}", file=sys.stderr)
    print(f"  unparseable XML : {n_fail}", file=sys.stderr)
    print(f"  no reference list: {n_empty}", file=sys.stderr)


if __name__ == "__main__":
    main()
