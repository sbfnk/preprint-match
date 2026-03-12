# Journal Prediction from medRxiv Preprints: Methodology and Results

## Problem

Given a medRxiv preprint, predict which journal it will ultimately be published in. The dataset has 35,366 labelled preprints across 4,402 journals. PLOS ONE, the largest class, has ~1,500 papers; many journals appear only once.

## Dataset

**Source**: [medRxiv](https://www.medrxiv.org/) preprints posted between June 2019 and March 2026 that were subsequently published in peer-reviewed journals.

- **Preprint metadata and full text**: [medRxiv API](https://api.medrxiv.org/) + JATS XML from the Cold Spring Harbor Laboratory S3 archive (`s3://biorxiv-src-monthly/Current_Content/`)
- **Publication destinations**: medRxiv API `published` field linking preprint DOIs to published DOIs
- **Journal names**: [Crossref API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) / Public Data File lookup from published DOIs

**Dataset statistics**:

| Metric | Value |
|---|---|
| Total labelled preprints | 35,366 |
| With full text | 27,105 (77%) |
| Unique journals | 4,402 |
| Journals with ≥10 papers | 316 |
| medRxiv categories | 51 |

**Preprocessing**: Journal names were normalised to resolve known case variants (e.g. "Plos One" → "PLOS ONE"). Full text was extracted from JATS XML using a custom parser, excluding reference sections and boilerplate.

## Evaluation Protocol

All experiments use a stratified 70/10/20 train/val/test split with random seed 42:

- **Training**: 24,844 papers — used for kNN index and classifier training
- **Validation**: 3,572 papers — used for hyperparameter tuning (ensemble alpha, classifier C, calibration temperature)
- **Test**: 6,950 papers — used for final evaluation only

**Stratification**: Papers are grouped by journal. For journals with ≥2 papers, the specified fraction goes to test (minimum 1 per set). Singleton journals are assigned to training only, since a single example cannot appear in both sets.

**Metrics**:

- **acc@k** (k=1, 5, 10): Fraction of test papers where the true journal appears in the top-k predictions. acc@1 = "got it exactly right"; acc@10 = "correct journal is somewhere in the top 10 list".
- **MRR** (Mean Reciprocal Rank): Average of 1/rank of the true journal. If the correct journal is ranked 1st, that paper contributes 1.0; if 5th, it contributes 0.2. Higher is better.
- **ECE** (Expected Calibration Error): How well predicted probabilities match observed frequencies. An ECE of 0.03 means predictions are off by about 3 percentage points on average.

**Per-tier breakdown**: Journals are assigned to frequency tiers based on training set counts:

| Tier | Definition | Test papers | Journals |
|---|---|---|---|
| Top-20 | 20 most common journals | 1,707 | 20 |
| Top-50 | Rank 21–50 | 712 | 30 |
| Mid-tail | ≥10 papers, not top-50 | 2,392 | 266 |
| Long-tail | <10 papers | 2,139 | ~4,100 |

**Filtered evaluation**: Results marked "min-10" restrict both predictions and evaluation to the 316 journals with ≥10 training papers (4,811 test papers). This reflects the production setting where we only serve predictions for journals with enough data to be meaningful.

## Experiment 1: Embedding Comparison (kNN Baseline)

### Method

k-Nearest Neighbours (k=20) with similarity-weighted voting. For each test paper, the k nearest training papers by cosine similarity are found, and their journals are ranked by summed similarity scores.

### Embeddings

Three embedding configurations were compared:

1. **[SPECTER2](https://huggingface.co/allenai/specter2) title+abstract** (768-dim): SPECTER2 base model with proximity adapter, encoding title + abstract only.
2. **SPECTER2 full-text** (768-dim): Same model, encoding title + abstract + full text. Long documents are split into overlapping chunks (stride=256 tokens), embedded via the CLS token, and mean-pooled.
3. **[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) full-text** (768-dim): Long-context embedding model (8,192 tokens).

### Results

| Embeddings | acc@1 | acc@5 | acc@10 | MRR |
|---|---|---|---|---|
| SPECTER2 title+abstract | 9.2% | 23.5% | 30.8% | 0.160 |
| **SPECTER2 full-text** | **9.9%** | **25.3%** | **32.9%** | **0.171** |
| nomic-v1.5 full-text | 8.5% | 20.6% | 27.0% | 0.142 |

SPECTER2 full-text is the best embedding. Full text provides a modest but consistent improvement over title+abstract alone. nomic underperformed despite its longer context window — domain-specific pre-training (SPECTER2 was trained on citation graphs) matters more than context length.

## Experiment 2: Contrastive Fine-tuning

### Method

Rather than fine-tuning the full [SPECTER2](https://huggingface.co/allenai/specter2) model (110M parameters), only the proximity adapter was fine-tuned (0.9M parameters, 0.8% of total). The base model stays frozen; only the adapter weights change.

**Training objective**: [InfoNCE](https://arxiv.org/abs/1807.03748) contrastive loss with in-batch negatives. For each paper in a batch, papers from the same journal serve as positives and all other papers in the batch as negatives.

**Hard negative mining**: Batches are constructed by grouping papers within the same medRxiv category (e.g. all Epidemiology papers together). This way, in-batch negatives are topically similar: the model has to distinguish between, say, a *Lancet Infectious Diseases* paper and a *Clinical Infectious Diseases* paper, not between an infectious disease paper and a cardiology paper.

**Training details**: 3 epochs, batch size 8 with up to 4 chunks per document (giving up to 7 in-batch negatives), learning rate 2e-5 with linear warmup, temperature 0.05. Training took ~3.5 hours on a single NVIDIA A40 GPU (48GB). After fine-tuning, all 35,366 papers were re-embedded (~3 hours).

### Results (kNN only)

| Embeddings | k | acc@1 | acc@5 | acc@10 | MRR |
|---|---|---|---|---|---|
| Original | 20 | 9.9% | 25.3% | 32.9% | 0.171 |
| Original | 50 | 10.5% | 28.1% | 37.2% | 0.189 |
| Fine-tuned | 20 | 11.5% | 28.1% | 36.4% | 0.192 |
| Fine-tuned (hard-neg) | 20 | 11.5% | 28.0% | 36.2% | 0.191 |
| **Fine-tuned (hard-neg)** | **50** | **12.3%** | **31.4%** | **41.6%** | **0.215** |

Fine-tuning gives ~1.5pp in acc@1 and ~3pp in acc@10. Higher k also helps. Hard negatives make little difference in kNN alone, but their benefit shows up in the ensemble (below).

## Experiment 3: Trained Classifier

### Method

Multinomial logistic regression on SPECTER2 embeddings (768-dim) + medRxiv category one-hot (52-dim) = 820 features. Regularisation strength C was tuned via grid search on the validation set over C ∈ {0.01, 0.1, 1.0, 10.0}.

### Results

| Embeddings | C | acc@1 | acc@10 | MRR | Notes |
|---|---|---|---|---|---|
| Fine-tuned | 1.0 | 12.0% | 39.7% | 0.207 | All 4,402 journals |
| Fine-tuned (hard-neg) | 10.0 | 13.1% | 41.5% | 0.220 | All 4,402 journals |

The classifier does well on frequent journals (37.8% acc@1 on top-20) but is near-zero on long-tail. C=10 was selected by validation grid search.

## Experiment 4: Ensemble

### Method

Score interpolation combining kNN similarity scores (converted to probabilities via softmax) with classifier softmax probabilities:

$$\text{score}(j) = \alpha \cdot s_{\text{kNN}}(j) + (1 - \alpha) \cdot s_{\text{clf}}(j)$$

Alpha was tuned via grid search on the validation set over α ∈ {0.0, 0.1, …, 1.0}.

### Results (min-10 filter, test set)

| Configuration | acc@1 | acc@5 | acc@10 | MRR | alpha | C |
|---|---|---|---|---|---|---|
| FT, k=20 | 18.2% | 44.6% | 58.5% | 0.309 | 0.1 | 1.0 |
| FT, k=20, C grid | 19.1% | 47.2% | 60.7% | 0.325 | 0.0 | 10.0 |
| Hard-neg, k=20, C grid | **20.0%** | **47.9%** | **61.4%** | **0.332** | 0.1 | 10.0 |
| Hard-neg, k=50, C grid | 19.7% | 47.4% | 61.9% | 0.331 | 0.1 | 10.0 |

**Per-tier (best: hard-neg ensemble, k=20, min-10)**:

| Tier | acc@1 | acc@5 | acc@10 | MRR | n |
|---|---|---|---|---|---|
| Top-20 | 35.6% | 69.9% | 82.4% | 0.512 | 1,707 |
| Top-50 | 17.1% | 46.1% | 62.9% | 0.314 | 712 |
| Mid-tail | 9.7% | 32.7% | 45.9% | 0.209 | 2,392 |

### Discussion

The best configuration uses hard-negative fine-tuned embeddings with an interpolation ensemble (alpha=0.1, C=10). C grid search was the single biggest improvement (+0.9pp acc@1); hard negatives added another +0.9pp. Together they take acc@1 from 18.2% to 20.0%.

Alpha=0.1 means 90% classifier, 10% kNN. With C=10 and enough training data, the classifier carries most of the weight. kNN still helps, especially for journals near the min-papers threshold.

Higher k (50 vs 20) marginally improves acc@10 (+0.5pp) but slightly hurts acc@1 (−0.3pp). More neighbours add noise to the top prediction but improve coverage.

## Calibration

Probabilities are calibrated using isotonic regression fitted on the validation set, followed by temperature scaling.

| Metric | Value |
|---|---|
| Temperature | 1.007 |
| ECE (uncalibrated, test) | 0.032 |
| ECE (calibrated, test) | 0.028 |
| Max confidence | 0.574 |
| Mean confidence | 0.155 |

The raw probabilities are already well-calibrated: temperature scaling barely changes them (T ≈ 1.0). The model never assigns more than ~57% to any single journal. ECE of 0.028 means predicted probabilities closely match observed frequencies.

## Summary

| Method | acc@1 | acc@10 | MRR | Notes |
|---|---|---|---|---|
| kNN (original, k=20) | 9.9% | 32.9% | 0.171 | All journals, baseline |
| kNN (hard-neg, k=50) | 12.3% | 41.6% | 0.215 | All journals, best kNN |
| Classifier (hard-neg, C=10) | 19.4% | 60.9% | 0.327 | min-10, classifier-only |
| **Ensemble (hard-neg, alpha=0.1)** | **20.0%** | **61.4%** | **0.332** | **min-10, best overall** |

The best method achieves 20.0% acc@1, 61.4% acc@10, and 0.332 MRR on the 316 journals with ≥10 training papers (covering 69% of test papers). For the top-20 journals, the correct journal appears in the top-10 list 82% of the time.

What mattered most, roughly in order: restricting to journals with ≥10 training papers (largest single effect), tuning classifier regularisation (C=10 vs C=1, +0.9pp acc@1), contrastive fine-tuning with hard negatives (+0.9pp acc@1), and using more neighbours in kNN (helps acc@10, slight cost to acc@1). Calibration required almost no correction (ECE 0.028).
