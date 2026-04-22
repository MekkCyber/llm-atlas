# Quality Filtering
*Depth — using a classifier or language model to score each document and drop low-quality ones.*

**TL;DR:** After deduplication, most of a raw web corpus is still low-quality — navigation menus, SEO spam, auto-generated filler, barely-coherent prose. Quality filtering applies a learned scorer (FastText classifier, small language model, or reference-judged classifier) to each document and keeps only those scoring above a threshold. The choice of *reference* for "high quality" is what makes or breaks the approach — DCLM (Li et al. 2024) and FineWeb-Edu (Penedo et al. 2024) showed that classifier-based filtering with a well-chosen reference pool substantially outperforms heuristic rules.

**Prereqs:** [tokenization](../fundamentals/_tokenization.md)
**Related:** [deduplication](deduplication.md), [decontamination](decontamination.md), [dolma](dolma.md), [_data-curation](_data-curation.md), [olmo-2 case study](../case-studies/olmo-2.md)

---

## What it is

Post-dedup web data is still heterogeneous in quality. You want the Wikipedia-prose tail of the distribution; you do not want the "click here to win" tail. Quality filtering is the pipeline step that applies a document-level score and drops documents below a threshold.

Three broad families of scorers, in roughly historical order:

| Approach | Scorer | Cost / doc | Example |
|---|---|---|---|
| Heuristic rules | Hand-coded filters (punctuation ratio, avg word length, banned phrases, perplexity on small LM) | Cheap | C4 (Raffel 2020), Gopher rules (Rae 2021) |
| Perplexity-based | Score documents with a small LM trained on high-quality reference (e.g. Wikipedia) | Moderate (needs LM forward pass) | CCNet (Wenzek 2019) |
| Classifier-based | FastText binary classifier trained on "high-quality" vs. "low-quality" reference pools | Very cheap (FastText is ~ms/doc) | DCLM, FineWeb-Edu, OLMo 2 |

Modern pipelines (2024 onward) predominantly use FastText classifiers with carefully chosen reference pools.

## How it works

### FastText classifier-based filtering

1. Pick a **high-quality reference pool** — e.g. Wikipedia + a specific curated corpus + LLM-judged "educational" pages. This choice is the main ML judgment in the pipeline.
2. Pick a **negative pool** — e.g. random Common Crawl samples (assumed mostly low-quality in aggregate).
3. Train a FastText binary classifier (Joulin et al., 2016) on `(reference positive, random negative)` labels. FastText is a shallow model: bag-of-n-grams → linear classifier. Trains in minutes on a single machine.
4. Score every document in the corpus with the classifier. Keep those above a threshold (e.g. top-decile, or > 0.5 classifier confidence).

Why FastText rather than a larger model: **cost**. A 10T-token corpus has billions of documents. A BERT-scale classifier would cost GPU-weeks; FastText is CPU-minutes. The accuracy gap is small when the classifier's job is "distinguish Wikipedia-like from random-CC-like" — a task that doesn't need deep semantics.

### DCLM's reference choice

DCLM (Li et al., 2024) was influential for making the choice of reference pool rigorous. They ran a **benchmark** where different candidate filters ("quality score A vs. B") were each used to build a corpus and train a fixed-architecture model; the downstream benchmark score ranked the filters. Their winning filter used OpenHermes + high-scored Reddit ELI5 as positives — not Wikipedia — because this better-targeted the *reasoning* distribution.

Takeaway: the reference pool defines what the filter is filtering *for*. Wikipedia-as-reference biases toward encyclopedic prose; code-rich-as-reference biases toward code-adjacent content; instruction-answer-as-reference biases toward explanatory writing. There is no universal "quality".

### FineWeb-Edu's variant

FineWeb-Edu (Penedo et al., 2024) used a different recipe: a small LM (Llama 3) was prompted to rate 500k sample pages on "educational value" 0–5; a classifier was then trained on those LLM-judged labels. Keep documents with score ≥ 3.

This distills LLM judgment into a cheap filter. It's effectively "ask the LLM what's educational, then learn to predict that at scale". The resulting corpus (FineWeb-Edu) reports strong benchmark performance on reasoning tasks compared to the parent FineWeb.

### Perplexity-based filtering (older)

CCNet (Wenzek et al., 2019) and similar pipelines scored each document's perplexity under a small LM trained on Wikipedia. Low perplexity (Wikipedia-like) passes the filter. Still used in some pipelines but largely superseded by classifier-based methods, which are cheaper and more flexible in what "quality" means.

### Heuristic rules (oldest)

C4 (Raffel et al., 2020), Gopher (Rae et al., 2021), and MassiveWeb applied dozens of hand-coded rules: minimum word count, punctuation ratio, presence of specific bad phrases, `<script>` tag count, etc. These are still used as a cheap first pass before the classifier — they remove obviously-broken documents (empty pages, raw HTML, binary data) for free.

## Why it matters

- **Most of the quality win is here.** Ablations in DCLM, FineWeb-Edu, and OLMo 2 all report that the quality-filter choice affects benchmark scores by several points — a larger effect than many architectural changes.
- **Cheap to iterate.** Swapping the reference pool and re-running the filter is hours/days of compute, not weeks. You can run many variants and pick the best with small-model proxy evals.
- **Composes with [deduplication](deduplication.md) and [decontamination](decontamination.md).** The three steps are orthogonal: dedup removes repetition; quality filters removes junk; decontamination removes eval leakage. Modern pipelines run all three.

## Gotchas & tricks

- **Reference pool is the whole game.** A poorly chosen reference silently biases the whole corpus. If you only train on Wikipedia-like docs you'll be bad at conversational or narrative tasks. DCLM's lesson: pick the reference to match the downstream target.
- **Classifier proxy evals can mislead.** A classifier that assigns Wikipedia-like text a high score tells you nothing about downstream task performance directly. Validate with small-model training runs on the filtered corpus, not just held-out classification accuracy.
- **Threshold choice is not neutral.** Keeping the top 10% vs. top 30% is a large corpus-size / quality trade. Too aggressive reduces token count below compute-optimal; too lenient keeps too much noise.
- **Don't apply web filters to code.** Code corpora need code-aware filters (e.g. parse validity, star count, test coverage), not FastText trained on prose references. Filtering GitHub with a Wikipedia-trained classifier drops most of the good code.
- **Filter once per domain.** Separate classifiers for web, code, academic, books. Shared classifiers over-normalize toward whichever domain dominates the reference pool.
- **Order matters: dedup before filtering.** FastText classifier scoring is cheap per doc, but scoring 10× the corpus because you didn't dedup first wastes throughput.
- **LLM-as-judge filters are expensive at scale.** FineWeb-Edu's 500k LLM-rated examples are a labeled-dataset step, not a per-document step. Applying an LLM to every document of a 10T corpus is infeasible without a distilled classifier.
- **Filters reflect the filter-maker's biases.** A reference pool emphasizing STEM will produce a STEM-heavy corpus; one emphasizing news will produce a news-heavy corpus. Document what went into your reference pool.

## Sources

- Paper: *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (C4) — Raffel et al., 2020 — heuristic-rule filtering as a baseline.
- Paper: *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* — Rae et al., 2021 — detailed heuristic filter catalog.
- Paper: *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data* — Wenzek et al., 2019 — perplexity-based filtering with KenLM.
- Paper: *Bag of Tricks for Efficient Text Classification (FastText)* — Joulin et al., 2016 — the classifier most quality filters use.
- Paper: *DataComp-LM: In search of the next generation of training sets for language models* — Li et al., 2024 — benchmarks filter choices, including classifier reference pool ablations.
- Paper / blog: *The FineWeb Datasets* — Penedo et al., 2024 — introduces FineWeb and FineWeb-Edu, including the LLM-judged-distillation classifier approach.
- Paper: *Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research* — Soldaini et al., 2024 — documents AI2's filtering pipeline.
- Paper: *2 OLMo 2 Furious* — AI2, 2024 — applies FastText-based classifier filtering in OLMo-Mix-1124.
