# MedRxiv Journal Prediction

Predict which journal a medRxiv preprint will be published in based on its content.

## Goal

Build a system where users can say "show me the 20 preprints from the last 2 months that most look like Lancet articles" - reverse-engineering journal gatekeeping to democratise quality assessment.

## Approach

1. **Embeddings + kNN** (baseline): Embed preprints, find nearest published papers, use their journal destinations
2. **Fine-tune small model**: If baseline insufficient, fine-tune Llama 3.1 8B or Mistral 7B
3. **LLM API**: Only if above fail (expensive)

## Data Sources

- **medRxiv preprints**: Full text XML from `s3://biorxiv-src-monthly/Current_Content/`
- **Publication destinations**: medRxiv API `published` field links preprint DOI → published DOI
- **Journal names**: Crossref API lookup from published DOIs
- **Citation counts**: Crossref public data file

## Data Not in Repo

Large data files (~3GB compressed) stored separately:
- `medrxiv_text.tar.bz2` - 70k preprint XMLs
- `crossref.tar.bz2` - 167M Crossref records
- `all_doi.tar.bz2` - DOI mappings

See [DATA_ACQUISITION.md](DATA_ACQUISITION.md) for instructions on obtaining the data.

## Scripts

- `extract_labeled_data.py` - Build labelled dataset from medRxiv + Crossref APIs

## Usage

```bash
# Extract labelled dataset (preprints with known journal destinations)
python3 extract_labeled_data.py --start-date 2024-01-01 --end-date 2024-06-30

# Full extraction for all available data
python3 extract_labeled_data.py --start-date 2020-01-01 --end-date 2024-12-31
```

## Status

- [x] Data pipeline understood
- [x] Extraction script working
- [ ] Full dataset extraction
- [ ] Embedding generation
- [ ] kNN baseline evaluation
- [ ] Fine-tuning (if needed)
