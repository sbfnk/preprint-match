#!/usr/bin/env python3
"""Fetch Sciety peer review data and build a DOI lookup for the webapp.

Downloads the Sciety docmaps index and extracts a compact mapping of
DOI → review groups for medRxiv/bioRxiv preprints.

Usage:
    python3 fetch_sciety.py --output predictions/sciety_reviews.json
"""

import argparse
import json
import re
import sys
import urllib.request


SCIETY_INDEX_URL = "https://sciety.org/docmaps/v1/index"


def fetch_sciety_reviews(output_path):
    """Download Sciety docmaps and extract DOI → review info."""
    print("Fetching Sciety docmaps index...", file=sys.stderr)
    try:
        with urllib.request.urlopen(SCIETY_INDEX_URL, timeout=60) as resp:
            data = json.load(resp)
    except Exception as e:
        print(f"Failed to fetch Sciety index: {e}", file=sys.stderr)
        # Write empty file so the webapp still works
        with open(output_path, "w") as f:
            json.dump({}, f)
        return

    articles = data.get("articles", [])
    print(f"  {len(articles)} docmaps total", file=sys.stderr)

    reviews = {}  # doi -> {groups: [...], url: ...}
    for docmap in articles:
        # Extract DOI from first step inputs
        first_step = docmap.get("first-step", "")
        steps = docmap.get("steps", {})
        step = steps.get(first_step, {})
        inputs = step.get("inputs", [])
        if not inputs:
            continue

        doi = inputs[0].get("doi", "")
        if not doi:
            continue

        # Only keep medRxiv/bioRxiv
        if not doi.startswith("10.1101/"):
            continue

        # Normalise: lowercase, strip version suffix
        doi = re.sub(r'v\d+$', '', doi.lower())

        group = docmap.get("publisher", {}).get("name", "")
        if not group:
            continue

        if doi not in reviews:
            reviews[doi] = {"groups": [], "url": f"https://sciety.org/articles/activity/{doi}"}
        if group not in reviews[doi]["groups"]:
            reviews[doi]["groups"].append(group)

    print(f"  {len(reviews)} medRxiv/bioRxiv papers with reviews", file=sys.stderr)

    with open(output_path, "w") as f:
        json.dump(reviews, f)
    print(f"  Written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Sciety peer review data")
    parser.add_argument("--output", default="predictions/sciety_reviews.json")
    args = parser.parse_args()
    fetch_sciety_reviews(args.output)
