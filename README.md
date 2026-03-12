# Preprint Match

Predict which journal a medRxiv preprint will be published in, with calibrated probabilities. Also supports the reverse: given a journal, find the preprints most likely to match it.

## Web app

```bash
python3 webapp.py --port 5000
```

Browse journals, explore paper predictions, and search across 54,000+ preprints and 501 journals. Light/dark theme follows system preference.

Also live at [preprints.epiforecasts.io](https://preprints.epiforecasts.io).

## How it works

Papers are embedded using SPECTER2 (full-text, adapter-only fine-tuning with InfoNCE loss and hard negative mining by medRxiv category). Predictions combine k=20 cosine-similarity kNN with multinomial logistic regression (C=10, tuned on validation set) on embeddings + category features. The ensemble interpolates 90% classifier / 10% kNN (alpha=0.1). Probabilities are calibrated with isotonic regression (ECE=0.028).

On the test set (6,950 papers, 316 journals with ≥10 training papers): gets the exact journal right 20% of the time, correct journal in the top 10 61% of the time. For the 20 most common journals, the top-10 list is correct 82% of the time.

## Dataset

35,366 labelled medRxiv preprints (2019–2026) across 4,402 journals. The webapp serves predictions for journals with at least 10 training papers (501 journals).

- **Preprint text**: medRxiv API + JATS XML from `s3://biorxiv-src-monthly/Current_Content/`
- **Publication destinations**: medRxiv API `published` field
- **Journal names**: Crossref API
- **Full text coverage**: 77% of papers (27,105 / 35,366)

Evaluation uses a 70/10/20 stratified train/val/test split. See [RESULTS.md](RESULTS.md) for detailed methodology and per-tier breakdowns.

## Usage

### Web interface

```bash
python3 webapp.py                    # http://localhost:5000
python3 webapp.py --port 8080        # custom port
```

### CLI tools

```bash
# Predict journals for a paper
python3 predict_journal.py --doi 10.1101/2021.12.28.21268468
python3 predict_journal.py --interactive

# Find preprints for a journal
python3 journal_filter.py "The Lancet Infectious Diseases" --top-k 20
python3 journal_filter.py --list-journals

# Recommend papers
python3 recommend.py --journals "eLife" "Nature Communications"
python3 recommend.py --papers 10.1101/2021.05.05.21256010
```

## Scripts

### Data acquisition

| Script | Purpose |
|---|---|
| `extract_labeled_data.py` | Fetch preprints from medRxiv API, look up journals via Crossref |
| `add_fulltext.py` | Match records to MECA XML files, extract body text |
| `parse_xml.py` | JATS XML parser |

### Modelling

| Script | Purpose |
|---|---|
| `generate_embeddings.py` | Generate SPECTER2 embeddings (title+abstract or full-text) |
| `finetune_embeddings.py` | Contrastive fine-tuning of SPECTER2 adapter |
| `regen_finetuned.py` | Re-embed with fine-tuned adapter |
| `evaluate_knn.py` | kNN baseline, stratified splits, per-tier metrics |
| `train_classifier.py` | Logistic regression / MLP classifier |
| `ensemble_predict.py` | kNN + classifier ensemble (interpolation / RRF) |
| `calibrate.py` | Calibration analysis, temperature scaling |

### Serving

| Script | Purpose |
|---|---|
| `predict_journal.py` | `JournalPredictor` class with save/load |
| `save_model.py` | Train and persist model to disk |
| `precompute.py` | Compute full probability matrix for webapp |
| `refresh.py` | Weekly refresh: fetch new preprints, embed, score |
| `webapp.py` | Flask web app |
| `journal_filter.py` | Journal-as-a-filter CLI tool |
| `recommend.py` | Paper recommendation (journal-based or embedding-based) |

## Reproducing from scratch

See [pipeline/README.md](pipeline/README.md) for step-by-step reproduction instructions, including HPC cluster setup.

## Data sources

See [DATA_ACQUISITION.md](DATA_ACQUISITION.md) for how to obtain the raw data (medRxiv XML, Crossref metadata).
