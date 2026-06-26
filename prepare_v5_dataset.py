#!/usr/bin/env python3
"""Prepare the v5 training dataset on the cluster.

Combines the freshly refreshed labels (current to today, from the /pubs
endpoint, abstract-only) with full text from the existing combined dataset,
and normalises journal names so the new /pubs records (which capitalise
prepositions, e.g. "Journal Of X") don't fragment journals already present
under their Crossref spelling ("Journal of X").

Inputs (in cwd):
  labeled_dataset_refreshed.json  - current labels (from data-latest release)
  labeled_dataset_combined.json   - older corpus WITH full_text
Output:
  labeled_dataset_v5.json         - current labels + full text, normalised

Usage: python3 prepare_v5_dataset.py
"""

import json
import sys


def main():
    print("Loading refreshed labels...", file=sys.stderr)
    with open("labeled_dataset_refreshed.json") as f:
        release = json.load(f)
    print(f"  {len(release)} records", file=sys.stderr)

    # Canonical journal spelling: prefer Crossref-sourced names (records that
    # carry a publisher). Map lowercased name -> canonical string.
    canon = {}
    for r in release:
        if r.get("publisher") and r.get("journal"):
            canon.setdefault(r["journal"].lower(), r["journal"])
    print(f"  {len(canon)} canonical journal names", file=sys.stderr)

    print("Loading combined dataset for full text (large)...", file=sys.stderr)
    with open("labeled_dataset_combined.json") as f:
        combined = json.load(f)
    ft = {r["preprint_doi"]: r["full_text"]
          for r in combined if r.get("full_text")}
    del combined
    print(f"  full_text available for {len(ft)} DOIs", file=sys.stderr)

    n_norm = n_ft = 0
    for r in release:
        c = canon.get((r.get("journal") or "").lower())
        if c and c != r["journal"]:
            r["journal"] = c
            n_norm += 1
        f = ft.get(r.get("preprint_doi"))
        if f:
            r["full_text"] = f
            n_ft += 1

    with open("labeled_dataset_v5.json", "w") as f:
        json.dump(release, f)

    print(f"\nWrote labeled_dataset_v5.json: {len(release)} records, "
          f"{n_ft} with full text, {n_norm} journal names normalised",
          file=sys.stderr)


if __name__ == "__main__":
    main()
