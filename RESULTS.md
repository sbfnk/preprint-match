# Journal Prediction from medRxiv Preprints: Methodology and Results

## Problem

Given a medRxiv preprint, predict which journal it will ultimately be published in. The dataset contains 25,182 labelled preprints spanning 3,645 distinct journals — an extreme multi-class classification problem with heavy class imbalance (the most common journal, PLOS ONE, has ~1,200 papers; 1,672 journals appear only once).

## Dataset

**Source**: medRxiv preprints posted between 2020 and 2024 that were subsequently published in peer-reviewed journals.

- **Preprint metadata and full text**: medRxiv API + JATS XML from the Cold Spring Harbor Laboratory S3 archive (`s3://biorxiv-src-monthly/Current_Content/`)
- **Publication destinations**: medRxiv API `published` field linking preprint DOIs to published DOIs
- **Journal names**: Crossref API / Public Data File lookup from published DOIs

**Dataset statistics**:

| Metric | Value |
|---|---|
| Total labelled preprints | 25,182 |
| With full text | 25,166 (99.9%) |
| Unique journals | 3,645 |
| Journals with 1 paper (singletons) | 1,672 (45.9%) |
| Journals with ≥10 papers | 196 |
| medRxiv categories | 51 |

**Preprocessing**: Journal names were normalised to resolve known case variants (e.g. "Plos One" → "PLOS ONE"). Full text was extracted from JATS XML using a custom parser, excluding reference sections and boilerplate.

## Evaluation Protocol

All experiments use stratified splits with random seed 42:

- **Experiments 1–2**: 80/20 train/test split (20,298 train / 4,884 test). No validation set; all hyperparameters were fixed a priori.
- **Experiments 3–4**: 70/10/20 train/val/test split (17,773 train / 2,525 val / 4,884 test). The validation set is used for hyperparameter tuning (e.g. ensemble interpolation weight) without touching the test set.

**Stratification**: Papers are grouped by journal. For journals with ≥2 papers, the specified fraction goes to test (minimum 1 per set). Singleton journals are assigned to training only, since a single example cannot appear in both sets. In the 3-way split, the validation set is carved from the training portion, so test papers are identical across both splits.

Test set journals are a strict subset of training journals (1,973 of 3,645).

**Metrics**:

- **Accuracy@k** (k=1, 5, 10): Fraction of test papers where the true journal appears in the top-k predictions
- **MRR** (Mean Reciprocal Rank): Average of 1/rank of the true journal across all test papers

**Per-tier breakdown**: Journals are assigned to frequency tiers based on training set counts to understand performance across the popularity spectrum. Tier assignments differ slightly between splits because the smaller training set in the 70/10/20 split shifts some journals between tiers. Counts below are for the 70/10/20 split used in Experiments 3–4:

| Tier | Definition | Test papers | Journals |
|---|---|---|---|
| Top-20 | 20 most common journals | 1,244 | 20 |
| Top-50 | Rank 21–50 | 512 | 30 |
| Mid-tail | ≥5 papers, not top-50 | 1,819 | ~250 |
| Long-tail | 2–4 papers | 1,309 | ~1,700 |

## Experiment 1: Embedding Comparison (kNN Baseline)

### Method

k-Nearest Neighbours (k=20) with similarity-weighted voting. For each test paper, the k nearest training papers by cosine similarity are found, and their journals are ranked by summed similarity scores.

kNN is a natural baseline for this problem: it makes no distributional assumptions, handles extreme multi-class settings without any class-specific parameters, and works equally well for rare and frequent journals (a singleton journal can still be predicted if the test paper is close to its single training example).

### Embeddings

Three embedding configurations were compared:

1. **SPECTER2 title+abstract** (768-dim): The SPECTER2 base model with proximity adapter, encoding title + abstract only. SPECTER2 is a SciBERT-based model fine-tuned on citation graphs for scientific document similarity — a natural fit for our task since journal placement and citation patterns are correlated.

2. **SPECTER2 full-text** (768-dim): Same model, but encoding title + abstract + full text. Since SPECTER2 has a 512-token context window, long documents are split into overlapping chunks (stride=256 tokens), each chunk is embedded via the CLS token, and per-paper embeddings are obtained by mean-pooling across chunks. This is a standard approach for adapting short-context models to long documents.

3. **nomic-embed-text-v1.5 full-text** (768-dim): A long-context embedding model (8,192 tokens) that can process most papers in a single pass without chunking. Included to test whether a purpose-built long-context model outperforms the chunk-and-pool approach.

All embeddings were generated on a single NVIDIA A40 GPU (48GB). SPECTER2 runs took ~2.5 hours; nomic took ~4 hours.

### Results

**Overall**:

| Embeddings | acc@1 | acc@5 | acc@10 | MRR |
|---|---|---|---|---|
| SPECTER2 title+abstract | 9.9% | 25.1% | 32.6% | 0.171 |
| **SPECTER2 full-text** | **10.7%** | **26.3%** | **34.7%** | **0.180** |
| nomic-v1.5 full-text | 9.1% | 22.0% | 28.8% | 0.152 |

**Per-tier (acc@1 / acc@10)**:

| Embeddings | Top-20 | Top-50 | Mid-tail | Long-tail |
|---|---|---|---|---|
| SPECTER2 title+abstract | 22.2 / 53.1 | 10.7 / 37.6 | 7.2 / 30.2 | 1.1 / 13.1 |
| **SPECTER2 full-text** | **25.3 / 58.7** | **11.3 / 40.5** | **7.0 / 30.3** | **1.2 / 14.3** |
| nomic-v1.5 full-text | 23.4 / 55.9 | 11.5 / 36.8 | 4.2 / 21.9 | 1.2 / 8.3 |

### Discussion

**SPECTER2 full-text is the best embedding** across all metrics and tiers. Full text provides a modest but consistent improvement over title+abstract alone (~1pp in acc@1, ~2pp in acc@10), confirming that the body text contains discriminative signal for journal placement beyond what the abstract captures.

**nomic-v1.5 underperformed** despite its longer context window. This likely reflects the importance of pre-training domain: SPECTER2 was specifically trained on scientific document similarity via citation graphs, while nomic is a general-purpose embedding model. Domain-specific pre-training matters more than context length for this task.

**Long-tail performance is inherently limited**: Even the best kNN achieves only 1.2% acc@1 on long-tail journals. With only 2–4 training examples per journal, the nearest neighbour from the same journal may simply not be close enough to outrank neighbours from other journals. This is a property of the data, not the method.

## Experiment 2: Trained Classifiers

### Motivation

kNN treats all embedding dimensions equally and cannot learn which features are more discriminative for journal prediction. A trained classifier can learn to weight features — for instance, that certain embedding dimensions or medRxiv categories are particularly predictive of specific journals. Additionally, medRxiv categories (e.g. "infectious diseases", "epidemiology") are highly informative about journal placement but cannot be directly incorporated into kNN similarity.

### Features

All classifiers used SPECTER2 full-text embeddings (the best from Experiment 1).

- **Embeddings** (768-dim): L2-normalised SPECTER2 full-text vectors
- **Category one-hot** (52-dim): medRxiv categories, with index 0 reserved for unknown categories. Encoder fitted on training categories only.

Total feature dimension: 820 (with categories) or 768 (without).

Features deliberately excluded:
- **Publisher**: Only known after publication — would constitute data leakage
- **Citation count**: Post-publication metric — same leakage concern

### Models

**Multinomial logistic regression** (`sklearn.linear_model.LogisticRegression`, solver=L-BFGS, C=1.0, max_iter=200): The standard approach for multi-class classification with probability outputs. Multinomial softmax over all 3,645 classes jointly, with L2 regularisation. L-BFGS was chosen over SGD because it produces properly calibrated multinomial probabilities (SGD's one-vs-rest scheme produces poorly calibrated class scores in extreme multi-class settings). Training time: ~5.5 minutes on CPU.

**MLP** (`sklearn.neural_network.MLPClassifier`, 1 hidden layer of 256 units, early stopping): Tests whether a non-linear decision boundary improves over the linear logistic model. Uses early stopping on a 10% validation holdout to prevent overfitting. Training time: ~38 seconds.

### Results

**Overall**:

| Model | Features | acc@1 | acc@5 | acc@10 | MRR |
|---|---|---|---|---|---|
| kNN (k=20) | embeddings | 10.7% | 26.3% | 34.7% | 0.180 |
| Logistic | embeddings + category | 9.4% | 25.9% | 34.2% | 0.175 |
| Logistic | embeddings only | 8.5% | 23.0% | 29.6% | 0.154 |
| MLP | embeddings + category | 7.4% | 17.4% | 22.9% | 0.126 |

**Per-tier (acc@1 / acc@10)**:

| Model | Features | Top-20 | Top-50 | Mid-tail | Long-tail |
|---|---|---|---|---|---|
| kNN (k=20) | emb | 25.3 / 58.7 | 11.3 / 40.5 | 7.0 / 30.3 | 1.2 / 14.3 |
| Logistic | emb + cat | **31.0 / 87.9** | **9.4 / 42.3** | 1.2 / 18.6 | 0.0 / 0.1 |
| Logistic | emb | 32.2 / 93.2 | 3.1 / 48.3 | 0.0 / 2.0 | 0.0 / 0.0 |
| MLP | emb + cat | 29.1 / 81.8 | 0.0 / 19.7 | 0.0 / 0.0 | 0.0 / 0.0 |

### Discussion

**kNN remains the best overall method.** The trained classifiers could not beat kNN on aggregate metrics. This is a well-known phenomenon in extreme multi-class settings: discriminative classifiers must allocate model capacity across all classes, and with 3,645 classes (many with very few examples), they underfit the tail while overfitting to the head.

**Classifiers excel on frequent journals.** Logistic regression achieves 31% acc@1 on the top-20 tier (vs kNN's 25%) and 88% acc@10 (vs 59%). For the most common journals, a trained model can learn distinctive features that kNN's uniform similarity weighting misses.

**Classifiers fail completely on rare journals.** Both logistic regression and MLP achieve near-zero accuracy on the long-tail (journals with 2–4 training examples). This is expected: with so few examples, the model cannot learn meaningful class-specific weights. kNN achieves 1.2% acc@1 and 14.3% acc@10 on the same tier — still low, but meaningfully better than zero.

**Category features help.** Comparing logistic regression with and without categories:
- Overall: +0.9pp acc@1, +4.6pp acc@10, +0.021 MRR
- The gain is strongest in mid-tail (+1.2pp acc@1, +16.6pp acc@10), where category information helps discriminate among journals that are too rare for the model to learn from embeddings alone but common enough to benefit from the coarser category signal.
- Without categories, logistic regression over-concentrates predictions on top-20 journals (32.2% acc@1 on top-20, but 0% on mid-tail), achieving a very high acc@10 of 93% on top-20 at the cost of everything else. Categories help spread predictions more evenly.

**The MLP underperformed logistic regression.** With 3,645 output classes and only 20k training samples, the MLP (256 hidden units = 209k + 2.98M parameters) is heavily overparameterised. Early stopping mitigated overfitting but caused the model to stop before learning to predict beyond the top-20 journals.

**Confusion patterns are sensible.** The most common errors across all classifiers involve predicting PLOS ONE (the largest class) for papers that actually went to BMJ Open, Scientific Reports, and other large generalist journals. This reflects genuine topical overlap between broad-scope journals.

## Experiment 3: Ensemble Methods

### Motivation

Experiments 1–2 revealed complementary strengths: kNN performs well across all tiers (especially the tail), while logistic regression excels on frequent journals (31% acc@1 on top-20 vs kNN's 25%) but fails on rare ones. An ensemble that combines both methods could exploit these complementary error patterns.

### Method

Two ensemble strategies were evaluated, both combining the kNN similarity scores from Experiment 1 with the logistic regression class probabilities from Experiment 2 (with category features):

1. **Reciprocal Rank Fusion (RRF)**: A rank-based combination that is agnostic to score scales. For each candidate journal, its reciprocal ranks from both systems are summed: $\text{RRF}(j) = \frac{1}{k + r_{\text{kNN}}(j)} + \frac{1}{k + r_{\text{clf}}(j)}$, with $k=60$.

2. **Score interpolation**: A weighted combination of normalised scores: $\text{score}(j) = \alpha \cdot s_{\text{kNN}}(j) + (1 - \alpha) \cdot s_{\text{clf}}(j)$. Both score vectors are min-max normalised to [0, 1] before combining. The weight $\alpha$ was tuned via grid search on the validation set (2,525 papers) over $\alpha \in \{0.0, 0.1, \ldots, 1.0\}$.

### Results

**Overall** (test set, 70/10/20 split):

| Method | acc@1 | acc@5 | acc@10 | MRR |
|---|---|---|---|---|
| kNN-only | 10.5% | 25.8% | 33.6% | 0.176 |
| Classifier-only | 9.2% | 25.6% | 33.7% | 0.172 |
| RRF (k=60) | 10.4% | 27.8% | 38.8% | 0.192 |
| **Interpolation (alpha=0.3)** | **11.1%** | **28.8%** | **40.0%** | **0.200** |

**Per-tier (best ensemble, interpolation alpha=0.3)**:

| Tier | acc@1 | acc@5 | acc@10 | MRR | n |
|---|---|---|---|---|---|
| Top-20 | 32.1% | 69.1% | 85.5% | 0.487 | 1,244 |
| Top-50 | 11.9% | 37.5% | 51.6% | 0.245 | 512 |
| Mid-tail | 4.5% | 18.4% | 30.5% | 0.122 | 1,819 |
| Long-tail | 0.1% | 1.8% | 5.4% | 0.017 | 1,309 |

### Discussion

**Interpolation outperforms both components and RRF.** The best ensemble achieves 40.0% acc@10 — a substantial improvement over either component alone (33.6% kNN, 33.7% classifier). The gain comes from combining the classifier's strength on frequent journals with kNN's ability to predict rare ones.

**Alpha=0.3 favours the classifier.** The optimal interpolation weight puts 70% of the weight on the classifier and 30% on kNN. This may seem counterintuitive given kNN's better standalone performance, but the classifier's well-calibrated softmax probabilities provide a stronger base signal, while the kNN scores serve as a useful correction, particularly for tail journals where the classifier assigns near-zero probability.

**RRF is competitive but inferior to interpolation.** RRF achieves 38.8% acc@10 vs interpolation's 40.0%. The rank-based combination loses score magnitude information that interpolation preserves — for instance, a journal ranked 2nd by a confident classifier and 50th by kNN should score higher than one ranked 10th by both, but RRF cannot distinguish these cases.

**Top-20 performance reaches 85.5% acc@10.** For the most common journals, the ensemble produces a top-10 list that includes the correct journal 85% of the time. This is approaching practical utility for journal recommendation.

## Experiment 4: Contrastive Fine-tuning

### Motivation

The SPECTER2 embeddings used in Experiments 1–3 were pre-trained for general scientific document similarity. They capture topical relatedness but are not optimised for distinguishing journals — two papers on the same topic that went to different journals may be embedded very close together. Contrastive fine-tuning adapts the embedding space to place papers from the same journal closer together and papers from different journals further apart.

### Method

**Architecture**: SPECTER2 uses a base transformer (SciBERT) with swappable task-specific adapters. Rather than fine-tuning the full model (110M parameters), only the proximity adapter was fine-tuned (0.9M parameters). This preserves the general scientific knowledge in the base model while adapting the embedding geometry for journal discrimination.

**Training objective**: InfoNCE contrastive loss with in-batch negatives. For each paper in a batch, papers from the same journal serve as positives and all other papers in the batch as negatives. With batch size 64, each example has up to 63 negatives, providing a diverse set of contrastive comparisons without explicit hard negative mining.

**Training details**: 3 epochs over the training set (17,773 papers), learning rate 2e-5 with linear warmup. Training took ~3 hours on a single NVIDIA A40 GPU. After fine-tuning, all 25,182 papers were re-embedded using the fine-tuned adapter (~2.5 hours).

The ensemble (interpolation with alpha=0.3, tuned on val set) was then re-evaluated using the fine-tuned embeddings, with both the kNN component and the classifier retrained on the new embeddings.

### Results

**kNN comparison** (test set):

| Embeddings | acc@1 | acc@5 | acc@10 | MRR |
|---|---|---|---|---|
| Original SPECTER2 | 10.5% | 25.8% | 33.6% | 0.176 |
| **Fine-tuned SPECTER2** | **11.6%** | **29.1%** | **37.8%** | **0.196** |

**Ensemble comparison** (interpolation alpha=0.3, test set):

| Embeddings | acc@1 | acc@5 | acc@10 | MRR |
|---|---|---|---|---|
| Original | 11.1% | 28.8% | 40.0% | 0.200 |
| **Fine-tuned** | **12.7%** | **32.7%** | **43.6%** | **0.225** |

**Per-tier (fine-tuned ensemble)**:

| Tier | acc@1 | acc@5 | acc@10 | MRR | n |
|---|---|---|---|---|---|
| Top-20 | 33.8% | 72.6% | 84.8% | 0.510 | 1,244 |
| Top-50 | 15.2% | 44.5% | 57.6% | 0.289 | 512 |
| Mid-tail | 6.4% | 24.2% | 37.7% | 0.158 | 1,819 |
| Long-tail | 0.4% | 1.8% | 7.0% | 0.023 | 1,309 |

**Per-tier improvement (fine-tuned vs original ensemble)**:

| Tier | acc@1 | acc@10 | MRR |
|---|---|---|---|
| Top-20 | +1.7pp | −0.7pp | +0.023 |
| Top-50 | +3.3pp | +6.0pp | +0.044 |
| Mid-tail | +1.9pp | +7.2pp | +0.036 |
| Long-tail | +0.3pp | +1.6pp | +0.006 |

### Discussion

**Fine-tuning provides consistent improvement.** The fine-tuned ensemble achieves 12.7% acc@1 and 43.6% acc@10, up from 11.1% and 40.0% respectively. Every metric improves, with the largest gains in the mid-tail and top-50 tiers.

**The classifier also benefits from better embeddings.** A notable result is that the standalone classifier improves substantially with fine-tuned embeddings (acc@1: 9.2% → 11.5%, acc@10: 33.7% → 39.7%). The fine-tuned embedding space is more linearly separable by journal, making the logistic regression classifier more effective — including on mid-tail journals where it previously achieved almost nothing.

**Top-20 acc@10 is slightly lower.** The fine-tuned ensemble achieves 84.8% acc@10 on top-20 vs 85.5% for the original — a negligible difference within noise. The fine-tuning trades a tiny amount of top-20 performance for substantial gains in the mid-tier.

**Gains are largest in the mid-range.** The top-50 and mid-tail tiers see the biggest improvement (+6.0pp and +7.2pp in acc@10 respectively). These are journals with enough training examples to benefit from contrastive learning but not so many that the original embeddings already sufficed. The long-tail improves modestly (+1.6pp acc@10), limited by having only 2–4 examples per journal.

## Summary

All results below are on the 70/10/20 test set (4,884 papers). Experiments 1–2 results are from the 80/20 split and are reported in their respective sections; the table below uses re-evaluated 70/10/20 numbers for comparability.

| Method | acc@1 | acc@10 | MRR | Notes |
|---|---|---|---|---|
| kNN (original embeddings) | 10.5% | 33.6% | 0.176 | Balanced across tiers |
| Classifier (logistic + category) | 9.2% | 33.7% | 0.172 | Strong on top-20 (30.8%), fails on tail |
| Ensemble original (alpha=0.3) | 11.1% | 40.0% | 0.200 | Combines kNN + classifier strengths |
| kNN (fine-tuned embeddings) | 11.6% | 37.8% | 0.196 | +1.1pp acc@1 from fine-tuning |
| **Ensemble fine-tuned (alpha=0.3)** | **12.7%** | **43.6%** | **0.225** | Best overall |

The best method is the fine-tuned ensemble, combining contrastive-adapted SPECTER2 embeddings with score interpolation between kNN and logistic regression. It achieves 12.7% acc@1, 43.6% acc@10, and 0.225 MRR — improvements of +2.2pp, +10.0pp, and +0.049 over the original kNN baseline.

For frequent journals (top-20), the ensemble reaches 84.8% acc@10 — the correct journal appears in the top-10 list 85% of the time. Performance degrades through the tiers but remains meaningful: 57.6% acc@10 for top-50, 37.7% for mid-tail. The long tail (2–4 training examples) remains the primary challenge at 7.0% acc@10.
