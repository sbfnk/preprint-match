<!-- Title to be decided. -->

Outline at one sentence per paragraph, per the lab manual.

## Research questions

1. How predictable is the journal a preprint ends up in, from the manuscript alone, across a large
   journal space?
2. Which parts of a manuscript does the prediction depend on: title, abstract, body text, section
   headings, author list, reference list?
3. Is placement predicted by the science or by its presentation and provenance?

## Background

1. Journals sort research by scope, standards and audience, and the sorting is slow: a manuscript is
   submitted, rejected, reformatted and resubmitted, often several times, and each round spends
   reviewer time on the same question of whether the paper belongs here.
2. Preprints removed the delay in sharing but not the delay in sorting, so work now appears
   immediately and then still spends months finding a venue.
3. Around twenty journal recommenders are already available, covering anything from a single
   publisher's titles to the whole of PubMed, but they return similarity scores rather than
   probabilities and they are not evaluated against where papers actually went.[^entrup]
4. Nicholson and colleagues come closest, predicting journals for bioRxiv preprints from real
   preprint-publication pairs, but they do so as a secondary demonstration inside a study of
   linguistic change, and report only that two classifiers beat a random baseline, stating neither an
   accuracy nor the size of the candidate set.[^nicholson]
5. How predictable placement is therefore remains unmeasured, and what such a model responds to has
   not been asked.
6. The answer matters in both directions: if placement is largely fixed by the manuscript, the
   submission cascade is an expensive way to rediscover something already settled; if it is not, the
   unexplained part is where editorial judgement, author ambition and prestige act, and its size is
   worth knowing.
7. Either way the measurement bears on reviewer workload, desk rejection, cascade routing between
   journals, and how much a journal name tells you about a paper beyond what the paper says itself.

## Methods

### Data

1. Preprints and their metadata come from the medRxiv and bioRxiv APIs, publication destinations
   from the same APIs' published-DOI field with journal names resolved through Crossref, and full
   text from the Cold Spring Harbor MECA archives as JATS XML.
2. The labelled set is 186,824 preprints posted since June 2019 that were later published, 95.8% of
   them with full text, and the analysis uses the 1,484 journals with at least ten training papers.
3. We split 70/10/20 into training, validation and test, stratified by journal with seed 42, so every
   journal with two or more papers appears in both training and test.

### Model

4. Papers are embedded with SPECTER2, a document encoder pre-trained on citation graphs, whose
   proximity adapter we fine-tune contrastively so that papers published in the same journal sit
   closer together; the encoder reads title, abstract and body in 512-token chunks, mean-pooled into
   one vector.
5. Predictions interpolate a k-nearest-neighbour lookup with a multinomial logistic regression,
   calibrated by temperature scaling and isotonic regression, and reach 21.7% acc@1 and 65.5% acc@10
   on the 33,031 held-out papers.

### Experiments

6. Removals ask what the prediction loses without a component: body text, section headings, abstract,
   title, and body truncated to *n* chunks.
7. Additions ask whether a component the model has never seen would help, namely the author list and
   the reference list, neither of which currently reaches it.
8. Every condition re-embeds the same 1,500 held-out papers and scores them through the fitted model,
   with acc@1, acc@10 and MRR as primary outcomes and top-1 agreement, top-5 overlap, rank shift of
   the true journal and cosine between probability vectors as secondary ones.
9. Four such ablations already exist and were run before this plan, so we report them as exploratory:
   headings cost 0.5pp acc@1, cited journals added to the classifier gain 0.2pp, removing body text
   costs 7.7pp acc@1 and 9.3pp acc@10, and a TF-IDF proxy misestimated the first two by 3x and 10x.
10. These are inference-time interventions on an adapter fine-tuned for one input shape, so a removal
    mixes lost signal with distribution shift and every result is a bound rather than an estimate,
    which is why we fine-tune dedicated adapters for the two conditions where the distinction changes
    the interpretation, body text and authors.
11. We state in advance how the author result will be read: if author identity predicts venue
    substantially once the model is retrained to use it, that is evidence about networks and prestige
    rather than content, and we report it either way, a null result included.

## Results

1. Table 1 lists every condition as a row, with acc@1, acc@10, MRR and top-1 agreement as columns,
   ordered by effect size.
2. Figure 1 sketches the same as a forest-style plot of change in acc@10 against the unmodified
   model, with zero marked, so the ordering and the size of each effect read at a glance.
3. Figure 2 sketches predicted against observed placement frequency by journal, which shows whether
   the calibrated probabilities hold across the range rather than only on average.
4. From what we already know, body text should dominate, headings and references should sit near
   zero, and the authors are an open question.

## Timeline

1. The matrix on the fitted model is roughly 2 GPU-hours and should run within a week.
2. The two retrained conditions are roughly 80 GPU-hours across two to three weeks.
3. Drafting follows from this outline, with paragraphs assigned once the outline is agreed.

[^entrup]: Entrup, Ewerth and Hoppe, A Comparison of Automated Journal Recommender Systems, TIB
    Leibniz Information Centre for Science and Technology, 2023.

[^nicholson]: Nicholson et al., Examining linguistic shifts between preprints and publications, PLOS
    Biology, 2022.
