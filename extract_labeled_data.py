#!/usr/bin/env python3
"""
Extract labeled dataset: preprints -> published journal destinations

Supports medRxiv and bioRxiv (same API, same structure).

Data flow:
1. Fetch preprints from medRxiv/bioRxiv API
2. Filter to those with published DOIs
3. Look up journal names from Crossref API
4. Save labeled dataset

Usage:
  python3 extract_labeled_data.py --server medrxiv [--start-date ...] [--output ...]
  python3 extract_labeled_data.py --server biorxiv [--start-date ...] [--output ...]
  python3 extract_labeled_data.py --server both   [--start-date ...] [--output ...]
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

def fetch_preprints(start_date: str, end_date: str, server: str = "medrxiv",
                    max_records: Optional[int] = None) -> list:
    """Fetch preprints from medRxiv or bioRxiv API with pagination."""
    if server not in ("medrxiv", "biorxiv"):
        raise ValueError(f"Unknown server: {server}")

    preprints = []
    cursor = 0
    batch_size = 100  # API returns max 100 per request

    while True:
        url = f"https://api.biorxiv.org/details/{server}/{start_date}/{end_date}/{cursor}"
        print(f"Fetching {server} cursor={cursor}...", file=sys.stderr)

        data = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = json.load(response)
                break
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                print(f"Attempt {attempt + 1}/5 failed: {e}", file=sys.stderr)
                if attempt < 4:
                    time.sleep(2 ** attempt)
        if data is None:
            print(f"Failed to fetch cursor={cursor} after 5 attempts", file=sys.stderr)
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
        total = int(data.get('messages', [{}])[0].get('total', 0))
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

    # Filter to preprints with published DOIs, deduplicate by preprint DOI
    # (API returns all versions, we only need latest with published info)
    # Also filter by DOI year if doi_year_filter is set
    published_dict = {}
    for p in preprints:
        if p.get('published') and p['published'] != 'NA' and p['doi'] not in seen_dois:
            # Keep the entry (later versions overwrite earlier ones)
            published_dict[p['doi']] = (p['doi'], p['published'], p)

    published = list(published_dict.values())
    print(f"  Total unique published: {len(published)}", file=sys.stderr)

    print(f"Found {len(published)} published preprints to process", file=sys.stderr)

    import contextlib
    ctx = open(progress_file, 'a') if progress_file else contextlib.nullcontext()

    with ctx as progress_f:
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
                    'source': preprint.get('_source', 'medrxiv'),
                }
                labeled.append(record)

                if progress_f:
                    progress_f.write(json.dumps(record) + '\n')
                    progress_f.flush()

            # Rate limit: ~10 requests per second for polite pool
            time.sleep(0.1)

    return labeled

def main():
    parser = argparse.ArgumentParser(description='Extract labeled preprint dataset')
    parser.add_argument('--server', default='medrxiv',
                        choices=['medrxiv', 'biorxiv', 'both'],
                        help='Preprint server to fetch from')
    parser.add_argument('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-06-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='labeled_data.json', help='Output file')
    parser.add_argument('--max-preprints', type=int, help='Max preprints to fetch')
    parser.add_argument('--progress-file', default='labeled_progress.jsonl', help='Progress file for resuming')
    parser.add_argument('--doi-year', help='Filter to DOIs from this year (e.g., 2024)')
    args = parser.parse_args()

    servers = ['medrxiv', 'biorxiv'] if args.server == 'both' else [args.server]

    all_preprints = []
    for server in servers:
        print(f"Fetching {server} preprints from {args.start_date} to {args.end_date}...",
              file=sys.stderr)
        preprints = fetch_preprints(args.start_date, args.end_date, server,
                                   args.max_preprints)
        print(f"Fetched {len(preprints)} {server} preprints", file=sys.stderr)

        # Tag each record with its source
        for p in preprints:
            p['_source'] = server

        all_preprints.extend(preprints)

    preprints = all_preprints

    # Filter by DOI year if specified (DOI format: 10.1101/YYYY.MM.DD.XXXXXXXX)
    if args.doi_year:
        pattern = f"10.1101/{args.doi_year}"
        preprints = [p for p in preprints if p['doi'].startswith(pattern)]
        print(f"Filtered to {len(preprints)} preprints from {args.doi_year}", file=sys.stderr)

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
