#!/usr/bin/env python3
"""Generate compact papers_slim.json for the webapp Docker image.

Drops has_fulltext, truncates abstracts, writes compact JSON.
"""
import json
import sys
from pathlib import Path

MAX_ABSTRACT = 500  # characters; enough for the fold-out preview


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("predictions/papers.json")
    dst = src.parent / "papers_slim.json"

    with open(src) as f:
        papers = json.load(f)

    slim = []
    for p in papers:
        entry = {
            "doi": p["doi"],
            "title": p.get("title", ""),
            "category": p.get("category", ""),
            "date": p.get("date", ""),
            "authors": p.get("authors", ""),
        }
        abstract = p.get("abstract", "")
        if len(abstract) > MAX_ABSTRACT:
            abstract = abstract[:MAX_ABSTRACT].rsplit(" ", 1)[0] + "..."
        entry["abstract"] = abstract
        slim.append(entry)

    with open(dst, "w") as f:
        json.dump(slim, f, separators=(",", ":"))

    print(f"{src} ({len(papers)} papers, {src.stat().st_size/1e6:.0f}MB) "
          f"-> {dst} ({dst.stat().st_size/1e6:.0f}MB)")


if __name__ == "__main__":
    main()
