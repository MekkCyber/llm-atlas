# Attribute-guided genre expansion
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An SFT-data pipeline that scales creative-writing corpora beyond story-only distributions by separating **thematic breadth** (from human-authored story prompts) from **genre-form control** (from manually curated per-genre attribute lists), then prompting a strong LLM for genre-faithful query/response pairs and quality-filtering. Chang et al. (2026) use it to build the 50K-example, 13-genre Multi-Genre Collection; fine-tunes on it beat both writing-specialised baselines and models trained on prior single-genre corpora on OOD writing benchmarks.

**Prereqs:** [_data-curation.md](./_data-curation.md), [quality-filtering.md](./quality-filtering.md)
**Related:** [dolma.md](./dolma.md)

---

## What it is

A synthetic-data recipe for a narrow domain (creative writing) where the base corpus is skewed to one form (short stories). Instead of scaling short-story data, the recipe **expands along a genre axis** while re-using existing human-authored prompts for thematic diversity.

Two orthogonal control knobs:

- **Prompt (theme)** — sampled from a large pool of human-authored creative-writing prompts. Provides *what to write about*.
- **Genre attributes** — a hand-curated list per target genre (rap, lyrics, scripts, game design, character design, …) that enforces the genre's structural, stylistic, and formatting conventions. Provides *how to write it*.

## How it works

1. **Seed pool.** Collect a large set of human-authored creative-writing prompts covering diverse themes.
2. **Genre attribute lists.** For each of 13 target genres, hand-curate an attribute checklist (structural elements, stylistic conventions, format constraints).
3. **Generation.** Prompt a strong writer LLM: *"produce a {genre} response to this prompt satisfying attributes {A}"*. Cross-product of prompts × genres gives raw pairs.
4. **Quality filter.** Score generated pairs (attribute conformance, coherence, style-fit) and drop the failures. The paper's ablations show this filter is load-bearing — unfiltered data underperforms.
5. **Result.** The Multi-Genre Collection: 50K examples across 13 genres, used as an SFT corpus.

## Why it matters

- **Attacks the story-centric bias.** Prior open creative-writing corpora are dominated by short stories; models fine-tuned on them are weak on rap, lyrics, scripts, game copy, and other structured forms.
- **Genre-count ablation shows *scale on the right axis*.** More genres beats more story data — the axis that matters is genre coverage, not raw token count.
- **Beats writing-specialised baselines.** Fine-tunes on the collection outperform both base models and prior writing-specialised models on out-of-distribution writing benchmarks.

## Gotchas & tricks

- **Attribute-list quality is the ceiling.** A poorly written per-genre attribute list produces uniform-looking outputs. This step is hand-authored and hard to skip.
- **Quality filter must include *attribute conformance*.** Standard "fluency + coherence" filters miss the whole point — a well-written short story labelled as "rap" passes fluency and fails the recipe.
- **Prompt/genre coverage bias.** If certain prompts are only ever paired with certain genres in the sampled cross-product, models can conflate theme with genre.
- **Requires a strong writer LLM.** The generator's ceiling caps the collection's quality; using a weak generator produces cheap volume but low-quality style.
- **English-only in the paper.** Applying the recipe to other languages requires per-language attribute curation from scratch.

## Sources

- Paper: *Scaling Creative Writing Beyond Story-Centric Data with Attribute-Guided Genre Expansion* — Chang et al., LG AI Research / Chung-Ang University, 2026 — [arXiv 2608.13947](https://arxiv.org/abs/2608.13947) — introduces the pipeline and the 50K Multi-Genre Collection.
