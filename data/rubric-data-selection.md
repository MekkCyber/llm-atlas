# Rubric-Anchored Data Selection

*Depth — discover per-source quality rubrics from the data itself, then distill the rubric into a fast scorer for full-corpus filtering.*

**TL;DR:** Mid-training data selection is awkward because the goal is downstream capability (post-SFT performance), the objective is pretraining-style (next-token), and the sources are heterogeneous (code repos, textbooks, Q&A, synthetic). Fixed cross-source rubrics underfit; model-based quality scores overfit. MIRA (Beihang / IQuest / SJTU 2026) discovers a *per-source-group* rubric via self-anchored induction on a seed sample, then distills the rubric judgments into a scalable student scorer that filters the full corpus.

**Prereqs:** [../pre-training/mid-training.md](../pre-training/mid-training.md), [quality-filtering.md](./quality-filtering.md)
**Related:** [_data-curation.md](./_data-curation.md), [decontamination.md](./decontamination.md), [deduplication.md](./deduplication.md), [dolma.md](./dolma.md)

---

## What it is

Mid-training data selection sits between two failure modes:

| Method class | Strength | Weakness |
| --- | --- | --- |
| **Model-based scorers** (perplexity gap, classifier vs ref model) | Scales to full corpus | Implicit signal — opaque, hard to inspect |
| **Semantic / rubric scorers** (LLM-as-judge with fixed prompt) | Explicit, inspectable | Assumes a fixed rubric — breaks across heterogeneous sources |

MIRA addresses the mismatch by making **rubric construction part of selection**: the rubric is discovered per source group, then distilled into a student scorer. Three stages:

1. **Source grouping.** Cluster the mid-training corpus into source groups by origin / format (code-repo, textbook-style, Q&A, synthetic-distillation, etc.).
2. **Self-anchored rubric induction.** For each group, derive a small rubric (a short list of "this source is valuable when…" criteria) from a seed sample of the data plus held-out downstream signal.
3. **Student-scorer distillation.** Score the seed under the rubric with a strong judge model; train a small student scorer to extrapolate those judgments to the full corpus.

## How it works

```
groups = source_cluster(corpus)
for g in groups:
    seed = sample(g, N=small)
    rubric_g = self_anchored_rubric_induction(seed, downstream_probes)
    seed_scores = judge_with_rubric(seed, rubric_g)
    scorer_g = train_student_scorer(seed, seed_scores)
    filtered_g = [x for x in g if scorer_g(x) > τ_g]
mid_training_data = union(filtered_g for g in groups)
```

Two design choices that matter:

- **Source-aware threshold $\tau_g$.** Code data and textbook data have different score distributions; per-group thresholds prevent one group from dominating the kept set.
- **Token-budget matching.** MIRA's headline result is matching the full-corpus run with half the tokens — meaning the budget, not just the threshold, is the relevant control knob.

## Why it matters

- **Mid-training is now a real stage.** R1, Kimi, OLMo 2, Qwen 2.5 all have a mid-training stage shaped toward downstream capabilities. The data-selection problem there is *different* from pretraining QC (which is mostly about deduplication and toxicity) and from SFT data quality (which is mostly about format and correctness).
- **Source-awareness > universal rubric.** A code repo is good when it has small, well-commented functions; a textbook is good when it has crisp definitions and worked examples. Forcing one rubric on both wastes signal. MIRA is the first principled instance of source-adaptive rubric induction.
- **Compute-positive.** Half the tokens, same performance — meaning the rubric pays for itself in mid-training compute saved.

## Gotchas & tricks

- **Seed sample size matters.** Too small: rubric overfits; too large: induction expensive. MIRA uses a moderate seed (low thousands per group).
- **Student scorer must generalize.** The student scorer is the only thing that runs at full-corpus scale; if it doesn't generalize from the judge-scored seed to the rest of the source group, filtering is noisy. Validate scorer transfer on held-out chunks.
- **Composes with dedup / decontam.** Run rubric scoring on a deduplicated, decontaminated corpus — don't double-count near-duplicates.
- **Iterate the rubric.** Rubrics inducted on a seed can be revised after seeing which kept examples drove downstream gain. Treat the rubric as a living document.

## Sources

- *MIRA: Mid-training Rubric Anchoring for Source-Aware Data Selection* — Wang et al., Beihang / IQuest / SJTU / Langboat, 2026 — [arXiv:2605.30288](https://arxiv.org/abs/2605.30288) — primary source.
- *DSIR / DoReMi / DataComp-LM* — earlier model-based selection baselines.
- *FineWeb / FineWeb-Edu* — fixed-classifier quality filtering baselines.
