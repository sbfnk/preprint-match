<!-- Title to be decided. -->

## Research questions

1. How predictable is journal placement from manuscript content alone, across a large journal space?
2. Which parts of a manuscript does the model use: title, abstract, body text, section headings,
   author list, reference list?
3. Is placement predicted by the science or by its presentation and provenance?

## Background

Journals are filters. They take a stream of research and select what fits their scope, standards and
audience. The filtering is slow and opaque: a manuscript is submitted, rejected, reformatted and
resubmitted, often several times, and each round consumes reviewer time to answer much the same
question of whether the paper belongs here. Preprints removed the delay in sharing but not the delay
in sorting, so work now appears immediately and then still spends months finding a venue.

How much of that sorting is settled by the manuscript itself is not known. If placement is largely
predictable from content, the submission cascade is an expensive way to rediscover something already
determined when the paper was written, and both authors and editors could be told the answer up
front. If it is not predictable, the unexplained part is where editorial judgment, author ambition,
prestige and chance operate, and its size is worth knowing. Either way the measurement bears on
peer review workload, on desk rejection, on cascade routing between journals, and on how much
information a journal name conveys about a paper beyond what the paper says itself.

Journal recommenders are not new. Entrup et al. survey twenty available systems, roughly a third of
them publisher-scoped and the rest covering a field or all journals, several of which already show
the user which similar articles produced a recommendation.[^entrup] What those systems compute is a
similarity score: BM25 averaged over matching articles for Elsevier's, summed and normalised Lucene
similarity for JANE. None reports a probability that a given journal will publish the paper, and the
same survey concludes that providers' lack of transparency about their methods means the results
cannot easily be interpreted.

So the gap is not recommendation itself. It is that nobody has published a calibrated, evaluated
measurement of how predictable placement is across a large journal space, or of what a model is
responding to when it predicts.

[^entrup]: Entrup, Ewerth and Hoppe, A Comparison of Automated Journal Recommender Systems, TIB
    Leibniz Information Centre for Science and Technology, 2023.

## Methods

### Data

Preprints and their metadata come from the medRxiv and bioRxiv APIs. Publication destinations come
from the same APIs' published-DOI field, with journal names resolved through Crossref. Full text
comes from the Cold Spring Harbor MECA archives as JATS XML, parsed to title, abstract and body
sections.

The labelled set is 186,824 preprints posted since June 2019 that were later published, spanning
7,203 journals. Analysis is restricted to the 1,484 journals with at least ten training papers,
since below that the model cannot learn a journal at all. Full text is available for 95.8% of the
corpus after a backfill on 2026-09-02. We use a stratified 70/10/20 train/validation/test split with
seed 42, grouping by journal so that every journal with two or more papers appears in both training
and test.

### Model

Papers are embedded with SPECTER2, a scientific document encoder pre-trained on citation graphs,
whose proximity adapter we fine-tune contrastively so that papers published in the same journal sit
closer together. Batches are grouped by preprint category so the negatives are topically similar.
The text given to the encoder is `title [SEP] abstract [SEP] body`, chunked at 512 tokens with
overlap and mean-pooled.

Predictions combine a k-nearest-neighbour lookup over training embeddings with a multinomial logistic
regression on the same embeddings plus a category feature, interpolated at alpha 0.1, then calibrated
by temperature scaling and isotonic regression. On the held-out test set of 33,031 papers this gives
21.7% acc@1 and 65.5% acc@10.

### Experiments

Two kinds of intervention on the fitted model.

Removals ask what the model loses without something: body text, section headings, abstract, title,
and body truncated to *n* chunks.

Additions ask whether something helps that the model has never seen: the author list and the
reference list. Neither reaches it today, because the encoder reads only `<body>` and the references
sit in `<back>`.

Each condition re-embeds the same 1,500 held-out papers and scores them through the fitted model.
Primary outcomes are acc@1, acc@10 and MRR. Secondary outcomes are top-1 agreement, top-5 overlap,
mean rank shift of the true journal, and cosine between the probability vectors.

Four such ablations already exist. We ran them without a plan, so they are exploratory and will be
reported that way. Removing section headings costs 0.5pp acc@1. Adding cited journals to the
classifier gains 0.2pp. Removing body text costs 7.7pp acc@1 and 9.3pp acc@10. A TF-IDF proxy
misestimated the first two by 3x and 10x, which is why every experiment here runs against the fitted
model rather than a stand-in.

**Known confound.** These are inference-time interventions on an adapter fine-tuned for one input
shape. A removal mixes lost signal with distribution shift, and an addition supplies text the model
never trained on. The body-text result shows how large that mixture can be: it came out eleven times
the estimate made before fine-tuning. We therefore treat every inference result as a bound. For the
two conditions where the distinction changes the interpretation, body text and authors, we fine-tune
a dedicated adapter and measure again, at roughly 40 GPU-hours each.

**Pre-specified interpretation.** Authors is the condition with an external claim attached. If author
identity predicts venue substantially once the model is retrained to use it, that is evidence about
networks and prestige rather than about content, and we report it either way. A null result is
equally publishable.

## Results sketch

One table. Conditions as rows, acc@1 / acc@10 / MRR / top-1 agreement as columns, ordered by effect
size. From what we already know, body text should dominate, headings sit near zero, and references
near zero. Authors is unknown, and it is the one we most want to see.

## Timeline

The matrix on the fitted model is about 2 GPU-hours, so within a week. The two retrained conditions
are about 80 GPU-hours, so two to three weeks. Write-up after that.
