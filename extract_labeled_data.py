#!/usr/bin/env python3
"""
Extract labeled dataset: medRxiv preprints -> published journal destinations

Data flow:
1. Fetch preprints from medRxiv API (2024 H1 for pilot)
2. Filter to those with published DOIs
3. Look up journal names from Crossref API
4. Save labeled dataset

Usage: python3 extract_labeled_data.py [--start-date 2024-01-01] [--end-date 2024-06-30] [--output labeled_data.json]
"""

import json
import csv
import time
import urllib.request
import urllib.error
import argparse
from pathlib import Path
from typing import Optional
import sys

# Crossref API wants an email for polite pool (faster rate limits)
CROSSREF_EMAIL = "medrxiv-pilot@example.com"  # Replace with real email for production

def fetch_medrxiv_preprints(start_date: str, end_date: str, max_records: Optional[int] = None) -> list:
    """Fetch preprints from medRxiv API with pagination."""
    preprints = []
    cursor = 0
    batch_size = 100  # API returns max 100 per request

    while True:
        url = f"https://api.medrxiv.org/details/medrxiv/{start_date}/{end_date}?cursor={cursor}"
        print(f"Fetching medRxiv cursor={cursor}...", file=sys.stderr)

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.load(response)
        except urllib.error.URLError as e:
            print(f"Error fetching medRxiv: {e}", file=sys.stderr)
            break

        batch = data.get('collection', [])
        if not batch:
            break

        preprints.extend(batch)
        cursor += len(batch)

        # Check if we've hit the limit
        if max_records and len(preprints) >= max_records:
            preprints = preprints[:max_records]
            break

        # Check if we've fetched all
        total = data.get('messages', [{}])[0].get('total', 0)
        if cursor >= total:
            break

        time.sleep(0.5)  # Be polite to the API

    return preprints

def lookup_journal_crossref(doi: str) -> Optional[dict]:
    """Look up journal name and metadata from Crossref API."""
    # Clean DOI
    doi = doi.strip()
    if doi.startswith('http'):
        doi = doi.split('doi.org/')[-1]

    url = f"https://api.crossref.org/works/{urllib.request.quote(doi, safe='')}"
    headers = {
        'User-Agent': f'MedRxivPilot/1.0 (mailto:{CROSSREF_EMAIL})',
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.load(response)
            msg = data.get('message', {})
            return {
                'journal': msg.get('container-title', [''])[0],
                'publisher': msg.get('publisher', ''),
                'type': msg.get('type', ''),
                'citation_count': msg.get('is-referenced-by-count', 0),
            }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"Crossref error for {doi}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error looking up {doi}: {e}", file=sys.stderr)
        return None

def build_labeled_dataset(preprints: list, progress_file: Optional[str] = None) -> list:
    """Build labeled dataset by looking up journal destinations."""
    labeled = []

    # Load progress if exists
    seen_dois = set()
    if progress_file and Path(progress_file).exists():
        with open(progress_file) as f:
            for line in f:
                record = json.loads(line)
                labeled.append(record)
                seen_dois.add(record['preprint_doi'])
        print(f"Loaded {len(labeled)} existing records", file=sys.stderr)

    # Filter to preprints with published DOIs
    published = [(p['doi'], p['published'], p) for p in preprints
                 if p.get('published') and p['published'] != 'NA'
                 and p['doi'] not in seen_dois]

    print(f"Found {len(published)} published preprints to process", file=sys.stderr)

    progress_f = open(progress_file, 'a') if progress_file else None

    for i, (preprint_doi, published_doi, preprint) in enumerate(published):
        if i % 50 == 0:
            print(f"Processing {i}/{len(published)}...", file=sys.stderr)

        # Look up journal from published DOI
        journal_info = lookup_journal_crossref(published_doi)

        if journal_info and journal_info['journal']:
            record = {
                'preprint_doi': preprint_doi,
                'published_doi': published_doi,
                'title': preprint.get('title', ''),
                'abstract': preprint.get('abstract', ''),
                'authors': preprint.get('authors', ''),
                'category': preprint.get('category', ''),
                'date': preprint.get('date', ''),
                'journal': journal_info['journal'],
                'publisher': journal_info['publisher'],
                'citation_count': journal_info['citation_count'],
            }
            labeled.append(record)

            if progress_f:
                progress_f.write(json.dumps(record) + '\n')
                progress_f.flush()

        # Rate limit: ~10 requests per second for polite pool
        time.sleep(0.1)

    if progress_f:
        progress_f.close()

    return labeled

def main():
    parser = argparse.ArgumentParser(description='Extract labeled medRxiv dataset')
    parser.add_argument('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-06-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='labeled_data.json', help='Output file')
    parser.add_argument('--max-preprints', type=int, help='Max preprints to fetch')
    parser.add_argument('--progress-file', default='labeled_progress.jsonl', help='Progress file for resuming')
    args = parser.parse_args()

    print(f"Fetching preprints from {args.start_date} to {args.end_date}...", file=sys.stderr)
    preprints = fetch_medrxiv_preprints(args.start_date, args.end_date, args.max_preprints)
    print(f"Fetched {len(preprints)} preprints", file=sys.stderr)

    # Count published
    published_count = sum(1 for p in preprints if p.get('published') and p['published'] != 'NA')
    print(f"Of which {published_count} have been published", file=sys.stderr)

    print("Building labeled dataset (this may take a while)...", file=sys.stderr)
    labeled = build_labeled_dataset(preprints, args.progress_file)

    # Save final output
    with open(args.output, 'w') as f:
        json.dump(labeled, f, indent=2)

    print(f"\nSaved {len(labeled)} labeled records to {args.output}", file=sys.stderr)

    # Print journal distribution
    journals = {}
    for r in labeled:
        j = r['journal']
        journals[j] = journals.get(j, 0) + 1

    print("\nTop 20 journals:", file=sys.stderr)
    for j, count in sorted(journals.items(), key=lambda x: -x[1])[:20]:
        print(f"  {count:4d}  {j}", file=sys.stderr)

if __name__ == '__main__':
    main()
