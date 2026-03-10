# Pipeline Documentation

This document describes the full pipeline for predicting which journal a medRxiv preprint will be published in. It covers every step from raw data acquisition through to serving predictions via a web application.

## Overview

The system works as follows:

1. **Data acquisition** — Fetch medRxiv preprints, match them to published journals via Crossref, and extract full text from MECA XML archives.
2. **Embedding** — Encode papers using SPECTER2 (title+abstract and full-text modes), then fine-tune the adapter via contrastive learning.
3. **Modelling** — Train a kNN baseline and logistic/CatBoost classifiers on embeddings, then combine them in an ensemble using interpolation fusion.
4. **Calibration** — Apply temperature scaling and assess calibration quality.
5. **Serving** — Precompute a probability matrix and serve it through a Flask web app, with CLI tools for filtering and recommendation.

### Architecture diagram

```
medRxiv API ──► Crossref API ──► labelled dataset (DOI → journal)
                                        │
MECA XML (S3) ──► full text extraction ─┘
                                        │
                                        ▼
                              SPECTER2 embeddings
                           (original + fine-tuned)
                                        │
                        ┌───────────────┼───────────────┐
                        ▼               ▼               ▼
                   cosine kNN    logistic reg.      CatBoost
                        │               │               │
                        └───────┬───────┘───────────────┘
                                ▼
                      interpolation ensemble (alpha)
                                │
                                ▼
                      temperature scaling / calibration
                                │
                                ▼
                      precomputed probability matrix
                                │
                        ┌───────┼───────┐
                        ▼       ▼       ▼
                     webapp   CLI     refresh
```

## Key Files

All scripts live in the project root.

### Data acquisition

| File | Purpose |
|---|---|
| `extract_labeled_data.py` | Fetch preprints from the medRxiv API, filter to those that have been published, look up journals via Crossref. Saves incrementally to a progress JSONL file. Args: `--start-date`, `--end-date`, `--output`, `--progress-file`. |
| `add_fulltext.py` | Match dataset records to XML files in `xml/`, extract body text, and add a `full_text` field to each record. |
| `parse_xml.py` | JATS XML parser. Key functions: `build_doi_index()`, `parse_jats_xml()`, `get_full_text_for_embedding()`. |

### Embeddings and fine-tuning

| File | Purpose |
|---|---|
| `generate_embeddings.py` | Generate SPECTER2 embeddings in title-abstract or full-text mode. Supports stride for long documents. |
| `finetune_embeddings.py` | Contrastive fine-tuning of the SPECTER2 adapter. Papers published in the same journal form positive pairs. Uses a 3-way stratified split. |
| `regen_finetuned.py` | Regenerate all embeddings using a fine-tuned adapter (run on the cluster after fine-tuning). |

### Evaluation and modelling

| File | Purpose |
|---|---|
| `evaluate_knn.py` | Stratified split (2-way or 3-way), cosine kNN evaluation. Includes `cosine_similarity_chunked` for large matrices. |
| `train_classifier.py` | Train logistic regression and CatBoost classifiers on embeddings + category features. |
| `ensemble_predict.py` | Combine kNN + classifier via interpolation (weighted alpha) or reciprocal rank fusion (RRF). Grid-searches alpha on the validation set. |
| `calibrate.py` | Reliability diagrams, expected calibration error (ECE), temperature scaling. Exports `ensemble_proba_matrix()` and `temperature_scale()` for downstream use. |

### Model persistence and serving

| File | Purpose |
|---|---|
| `predict_journal.py` | `JournalPredictor` class with `save()` / `load()` for model persistence. Used by the webapp and CLI tools. |
| `save_model.py` | One-time script to train the full pipeline and persist the model to `model/`. |
| `precompute.py` | Fetch all current preprints, embed them, and compute the full probability matrix for the webapp. |
| `refresh.py` | Weekly refresh pipeline: fetch new preprints, embed, and score against the saved model. |
| `webapp.py` | Flask web application serving journal predictions. |

### CLI tools

| File | Purpose |
|---|---|
| `journal_filter.py` | Given a journal name, rank preprints by predicted probability. Supports fuzzy matching, `--list-journals`, and `--interactive` mode. Uses calibrated ensemble probabilities. |
| `recommend.py` | Paper recommendation with two modes: (1) journal-based — specify journals, find matching preprints; (2) embedding-based — provide example DOIs, find similar papers. |

## Reproducing from Scratch

### Prerequisites

- Python 3.10+ with dependencies installed (see `venv/`)
- Access to the LSHTM HPC cluster for GPU steps (fine-tuning, embedding generation)
- MECA XML files in `xml/` (approximately 70k files)

---

### Step 1: Build the labelled dataset

**Where:** local machine
**Time:** approximately 30 minutes if seeded from existing data; several hours from scratch (rate-limited by Crossref)
**Output:** `labeled_dataset_v2.json`, `labeled_progress_v2.jsonl`

If you have an existing dataset, seed the progress file first to avoid re-fetching:

```bash
python3 -c "
import json
d = json.load(open('labeled_dataset.json'))
f = open('labeled_progress_v2.jsonl', 'w')
[f.write(json.dumps(r) + '\n') for r in d]
f.close()
"
```

Then run the extraction:

```bash
python3 extract_labeled_data.py \
  --start-date 2019-06-01 --end-date 2026-03-09 \
  --output labeled_dataset_v2.json \
  --progress-file labeled_progress_v2.jsonl
```

The script fetches preprint metadata from the medRxiv API in batches, checks each DOI against Crossref for publication status, and writes results incrementally to the progress file. If interrupted, re-running will resume from where it left off.

---

### Step 2: Add full text

**Where:** local machine
**Time:** approximately 30 minutes
**Output:** `labeled_dataset.json` (approximately 1.3 GB with full text)

```bash
python3 add_fulltext.py \
  --input labeled_dataset_v2.json \
  --output labeled_dataset.json
```

This matches each record to its MECA XML file in `xml/`, parses the JATS body text, and adds a `full_text` field. Records without a matching XML file retain a null full-text field and fall back to title+abstract embeddings.

---

### Step 3: Generate baseline embeddings

**Where:** HPC cluster (GPU required)
**Time:** approximately 2–3 hours
**Output:** `embeddings/full-text-specter2/` (`.npy` files + metadata)

Sync scripts to the cluster:

```bash
rsync -av --exclude='.git' /Volumes/code/medrxiv/*.py hpclogin:~/medrxiv/
rsync -av /Volumes/code/medrxiv/labeled_dataset.json hpclogin:~/medrxiv/
```

On the cluster:

```bash
python3 generate_embeddings.py \
  --input labeled_dataset.json \
  --mode full-text --model specter2 \
  --output-dir embeddings/full-text-specter2 \
  --stride 256
```

This encodes each paper using SPECTER2. For full-text mode, long documents are split into overlapping windows (stride 256 tokens) and the embeddings are averaged. The output directory contains NumPy arrays and a metadata JSON mapping indices to DOIs.

---

### Step 4: Fine-tune SPECTER2

**Where:** HPC cluster (GPU required)
**Time:** approximately 3–4 hours
**Output:** `finetuned-specter2-v2/` (adapter weights + regenerated embeddings)

```bash
python3 finetune_embeddings.py \
  --input labeled_dataset.json \
  --output-dir finetuned-specter2-v2/ \
  --epochs 3 --batch-size 16 --lr 2e-5 \
  --val-size 0.1 --test-size 0.2
```

Contrastive learning: papers published in the same journal form positive pairs, others form negatives. Only the adapter layers are trained (the base SPECTER2 model is frozen). After fine-tuning, the script regenerates embeddings using the best adapter checkpoint. These typically give a 1–3 percentage point lift across all metrics.

If fine-tuning and embedding regeneration need to be run separately (e.g. to fit within SLURM time limits), use `finetune.sbatch` followed by `regen_embeddings.sbatch`.

---

### Step 5: Evaluation

**Where:** HPC cluster (CPU is sufficient for most steps)
**Time:** approximately 1 hour total

#### kNN baselines

```bash
# Original embeddings
python3 evaluate_knn.py \
  --embeddings-dir embeddings/full-text-specter2 \
  --val-size 0.1

# Fine-tuned embeddings
python3 evaluate_knn.py \
  --embeddings-dir finetuned-specter2-v2/embeddings \
  --val-size 0.1
```

#### Classifier training

```bash
python3 train_classifier.py \
  --embeddings-dir finetuned-specter2-v2/embeddings \
  --val-size 0.1
```

#### Ensemble

```bash
# Full ensemble with both interpolation and RRF
python3 ensemble_predict.py \
  --embeddings-dir finetuned-specter2-v2/embeddings \
  --val-size 0.1 --method both

# With min-papers=10 (restricts to journals with at least 10 training examples)
python3 ensemble_predict.py \
  --embeddings-dir finetuned-specter2-v2/embeddings \
  --val-size 0.1 --method interpolation --min-papers 10
```

The ensemble grid-searches over alpha (the interpolation weight between kNN and classifier scores) on the validation set and reports test-set metrics. Note the best alpha value for Step 6.

#### Calibration

```bash
python3 calibrate.py \
  --embeddings-dir finetuned-specter2-v2/embeddings
```

Produces reliability diagrams, ECE, and an optimal temperature scaling parameter.

#### Expected results

Results will vary depending on the dataset size. See [RESULTS.md](../RESULTS.md) for detailed methodology.

---

### Step 6: Save model and precompute predictions

**Where:** HPC cluster (GPU required for precompute)
**Time:** approximately 2 hours
**Output:** `model-v2/` (saved model), `predictions/` (probability matrix + metadata)

```bash
python3 save_model.py \
  --model-dir model-v2 \
  --embeddings-dir finetuned-specter2-v2/embeddings \
  --alpha <best_from_step5> --min-papers 10
```

Replace `<best_from_step5>` with the optimal alpha value from the ensemble grid search (currently 0.3 for the fine-tuned embeddings, 0.1 for min-papers=10).

Then precompute the full probability matrix for the webapp:

```bash
python3 precompute.py --skip-fetch --all \
  --model-dir model-v2 \
  --adapter-path finetuned-specter2-v2/best_adapter
```

Sync results back to local:

```bash
rsync -av hpclogin:~/medrxiv/model-v2/ /Volumes/code/medrxiv/model-v2/
rsync -av hpclogin:~/medrxiv/predictions/ /Volumes/code/medrxiv/predictions/
```

---

### Step 7: Launch the web application

**Where:** local machine
**Time:** starts in seconds

```bash
python3 webapp.py --predictions-dir predictions
```

The Flask app loads the precomputed probability matrix and serves journal predictions through a web interface (templates in `templates/`, static assets in `static/`).

---

## Weekly Refresh

To keep predictions up to date with new preprints:

```bash
python3 refresh.py --days 30 --model-dir model
```

This fetches preprints posted in the last 30 days, generates embeddings using the saved adapter, scores them against the persisted model, and updates the predictions directory. State is tracked in `refresh_state.json` to avoid reprocessing.

---

## HPC Cluster Details

| Setting | Value |
|---|---|
| Host | `hpclogin` (SSH alias) |
| GPU | 1 node (c21), 2x NVIDIA A40 (48 GB each) |
| CUDA | 12.8 |
| Conda env | `medrxiv` |
| Working directory | `~/medrxiv/` |
| SLURM time limit | 4 hours per job |

Sync local scripts to the cluster:

```bash
rsync -av --exclude='.git' /Volumes/code/medrxiv/*.py hpclogin:~/medrxiv/
```

For jobs exceeding the 4-hour limit, split fine-tuning and embedding regeneration into separate sbatch submissions (`finetune.sbatch` and `regen_embeddings.sbatch`).

---

## Data Files

All large data files are gitignored. Here is where everything lives:

| Path | Contents | Size |
|---|---|---|
| `labeled_dataset.json` | Full labelled dataset with full text | ~1.3 GB |
| `embeddings/` | Original SPECTER2 embeddings (title-abstract, full-text) | ~500 MB |
| `finetuned-specter2/` | Fine-tuned adapter weights + regenerated embeddings | ~600 MB |
| `model/` | Saved model (classifier, metadata, training embeddings) | ~200 MB |
| `predictions/` | Precomputed probability matrix + paper metadata | ~100 MB |
| `xml/` | MECA XML files | ~70k files |
| `*.jsonl` | Progress files for incremental data fetching | varies |

---

## Dataset Statistics (v2, March 2026)

- **Total labelled preprints:** 35,366
- **Journals:** 4,403 (316 with at least 15 papers served in webapp)
- **With full text:** 27,105 (77%)
- **Split:** 70/10/20 stratified (train/val/test)
- **Validation set** is used for hyperparameter tuning (alpha, temperature); **test set** for final reported metrics.
