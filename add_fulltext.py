#!/usr/bin/env python3
"""Add full text from MECA XML files to labelled dataset.

Reads a labelled dataset JSON, matches each paper's preprint DOI to an XML
file in xml/, extracts the body text, and writes back the enriched dataset.

Usage:
  python3 add_fulltext.py --input labeled_dataset_v2.json --output labeled_dataset.json
  python3 add_fulltext.py --input labeled_dataset_v2.json  # overwrite in place
"""

import json
import argparse
import sys
from pathlib import Path

from parse_xml import build_doi_index, parse_jats_xml, find_xml_by_doi, INDEX_FILE, XML_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Add full text from XML to labelled dataset")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument("--output", help="Output dataset JSON (default: overwrite input)")
    parser.add_argument("--xml-dir", default="xml", help="XML directory")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force rebuild of DOI→XML index")
    args = parser.parse_args()

    output = args.output or args.input

    # Load dataset
    with open(args.input) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} records from {args.input}", file=sys.stderr)

    # Load or build DOI index
    index_path = Path(INDEX_FILE)
    xml_dir = Path(args.xml_dir)

    if args.rebuild_index or not index_path.exists():
        print("Building DOI→XML index...", file=sys.stderr)
        index = build_doi_index(xml_dir)
        with open(index_path, "w") as f:
            json.dump(index, f)
        print(f"Index built: {len(index)} DOIs", file=sys.stderr)
    else:
        with open(index_path) as f:
            index = json.load(f)
        print(f"Loaded existing index: {len(index)} DOIs", file=sys.stderr)

    # Match and extract full text
    matched = 0
    already_had = 0
    errors = 0

    for i, record in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(dataset)} (matched {matched})...",
                  file=sys.stderr)

        # Skip if already has full text
        if record.get("full_text"):
            already_had += 1
            continue

        doi = record.get("preprint_doi", "")
        xml_file = find_xml_by_doi(doi, index)
        if xml_file is None:
            continue

        try:
            parsed = parse_jats_xml(xml_file)
            if parsed.get("body_text"):
                record["full_text"] = parsed["body_text"]
                record["xml_file"] = str(xml_file)
                matched += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error parsing {xml_file}: {e}", file=sys.stderr)

    # Save
    with open(output, "w") as f:
        json.dump(dataset, f, indent=2)

    total_with_text = sum(1 for r in dataset if r.get("full_text"))
    print(f"\nDone:", file=sys.stderr)
    print(f"  Records: {len(dataset)}", file=sys.stderr)
    print(f"  Already had full text: {already_had}", file=sys.stderr)
    print(f"  Newly matched: {matched}", file=sys.stderr)
    print(f"  Parse errors: {errors}", file=sys.stderr)
    print(f"  Total with full text: {total_with_text}/{len(dataset)}", file=sys.stderr)
    print(f"Saved to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
