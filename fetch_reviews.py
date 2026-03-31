#!/usr/bin/env python3
"""Fetch community evaluation data from Sciety, PCI, and PubPeer.

Downloads peer review / evaluation metadata and builds a compact DOI
lookup for the webapp.

Usage:
    python3 fetch_reviews.py --output predictions/community_reviews.json
"""

import argparse
import json
import re
import sys
import time
import urllib.request

SCIETY_INDEX_URL = "https://sciety.org/docmaps/v1/index"

PCI_COMMUNITIES = [
    "animsci", "archaeo", "ecology", "ecotoxenvchem", "evolbiol",
    "forestwoodsci", "genomics", "healthmovsci", "infections", "mcb",
    "microbiol", "networksci", "neuro", "nutrition", "orgstudies",
    "paleo", "plants", "psych", "rr", "statml", "zool",
]

# PubPeer batch size (they accept multiple DOIs per request)
PUBPEER_BATCH = 50


def _normalise_doi(doi):
    """Lowercase, strip version suffix."""
    return re.sub(r'v\d+$', '', doi.lower().strip())


def _add_review(reviews, doi, source, url):
    """Add a review source for a DOI."""
    if doi not in reviews:
        reviews[doi] = {"sources": [], "urls": {}}
    if source not in reviews[doi]["sources"]:
        reviews[doi]["sources"].append(source)
    reviews[doi]["urls"][source] = url


def fetch_sciety(reviews):
    """Fetch Sciety docmaps index."""
    print("Fetching Sciety...", file=sys.stderr)
    try:
        with urllib.request.urlopen(SCIETY_INDEX_URL, timeout=60) as resp:
            data = json.load(resp)
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        return

    articles = data.get("articles", [])
    count = 0
    for docmap in articles:
        first_step = docmap.get("first-step", "")
        step = docmap.get("steps", {}).get(first_step, {})
        inputs = step.get("inputs", [])
        if not inputs:
            continue

        doi = inputs[0].get("doi", "")
        if not doi or not doi.startswith("10.1101/"):
            continue

        doi = _normalise_doi(doi)
        group = docmap.get("publisher", {}).get("name", "")
        if not group:
            continue

        url = f"https://sciety.org/articles/activity/{doi}"
        _add_review(reviews, doi, f"Sciety ({group})", url)
        count += 1

    print(f"  {count} evaluations from Sciety", file=sys.stderr)


def fetch_pci(reviews):
    """Fetch Peer Community In recommendations across all communities."""
    print("Fetching PCI...", file=sys.stderr)
    count = 0
    for community in PCI_COMMUNITIES:
        url = f"https://{community}.peercommunityin.org/api/recommendations"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                recs = json.load(resp)
        except Exception as e:
            print(f"  {community}: failed ({e})", file=sys.stderr)
            continue

        for rec in recs:
            article_doi = rec.get("article", {}).get("doi", "")
            if not article_doi or not article_doi.startswith("10.1101/"):
                continue

            doi = _normalise_doi(article_doi)
            rec_url = rec.get("recommendation", {}).get("url", "")
            label = f"PCI {community.replace('evolbiol', 'Evol Biol').replace('mcb', 'MCB').replace('rr', 'Registered Reports')}"
            _add_review(reviews, doi, label, rec_url or url)
            count += 1

        time.sleep(0.3)

    print(f"  {count} recommendations from PCI", file=sys.stderr)


def fetch_pubpeer(reviews, dois):
    """Fetch PubPeer comment counts for a list of DOIs."""
    print("Fetching PubPeer...", file=sys.stderr)
    doi_list = [d for d in dois if d.startswith("10.1101/")]
    count = 0

    for i in range(0, len(doi_list), PUBPEER_BATCH):
        batch = doi_list[i:i + PUBPEER_BATCH]
        payload = json.dumps({"dois": batch}).encode()
        req = urllib.request.Request(
            "https://pubpeer.com/v3/publications?devkey=PubMedPython",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.load(resp)
        except Exception as e:
            print(f"  Batch {i}: failed ({e})", file=sys.stderr)
            continue

        for pub in result.get("feedbacks", []):
            n = pub.get("total_comments", 0)
            if n > 0:
                pub_doi = pub.get("id", "")
                if pub_doi:
                    doi = _normalise_doi(pub_doi)
                    url = pub.get("url", f"https://pubpeer.com/search?q={doi}")
                    _add_review(reviews, doi, "PubPeer", url)
                    count += 1

        if i + PUBPEER_BATCH < len(doi_list):
            time.sleep(0.5)

    print(f"  {count} papers with PubPeer comments", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch community evaluation data")
    parser.add_argument("--output", default="predictions/community_reviews.json")
    parser.add_argument("--papers", default="predictions/papers.json",
                        help="Papers file for PubPeer DOI list")
    parser.add_argument("--skip-pubpeer", action="store_true",
                        help="Skip PubPeer (slow for large paper sets)")
    args = parser.parse_args()

    reviews = {}

    fetch_sciety(reviews)
    fetch_pci(reviews)

    if not args.skip_pubpeer and args.papers:
        try:
            with open(args.papers) as f:
                papers = json.load(f)
            dois = [p["doi"] for p in papers]
            fetch_pubpeer(reviews, dois)
        except FileNotFoundError:
            print("  No papers file, skipping PubPeer", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(reviews, f)

    print(f"\nTotal: {len(reviews)} papers with community evaluations",
          file=sys.stderr)


if __name__ == "__main__":
    main()
