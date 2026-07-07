# DCVLM — DataComp for VLMs
*Depth — the shared benchmark for VLM data-curation strategies, and its "mixing beats filtering" headline.*

**TL;DR:** DCVLM is a benchmark that fixes the training pipeline and lets researchers swap the *data pipeline*, so filtering, mixing, formatting, and sampling strategies can be compared apples-to-apples on 1B–8B VLMs across 6.25B–200B token budgets. The corpus: 160 datasets, 6T multimodal tokens, partitioned into image-caption, interleaved documents, text-only, and instruction data. Headline finding: for VLMs, *data mixing is the primary lever, not filtering* — instruction-heavy mixtures scale better than caption-heavy ones, with gaps widening at larger scales. The provided DCVLM-Baseline hits 63.6% on the 33-task core suite (8B VLM, 200B tokens), a +5.4pp improvement over FineVision.

**Prereqs:** [_data-curation](_data-curation.md), [dolma](dolma.md)
**Related:** [quality-filtering](quality-filtering.md), [../multimodal/README.md](../multimodal/README.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

DataComp-1B pioneered the "fix model + compute, vary data" harness for CLIP-style image-text training. DCVLM ports the same idea to modern instruction-tuned VLMs, which have a materially different training story: multiple data *types* (caption, interleaved, text-only, instruction) with very different roles at different training stages.

The benchmark treats data as a portfolio problem. Participants receive four labeled data pools, a fixed VLM training pipeline, and a fixed evaluation harness (52 downstream benchmarks across 9 domains, aggregated into a 33-task core score). They tune:

- **Filtering** *within* each pool (quality thresholds, dedup, CLIP-score cutoffs, …).
- **Mixing** *across* pools (what fraction of training tokens comes from each type).
- **Formatting** (instruction templates, chat-template composition).
- **Sampling** (curriculum, temperature over sources).

Everything else — architecture, optimizer, LR schedule, evaluation — is frozen.

## How it works

**The corpus.** 6T tokens spanning:

| Pool | Content | Role |
| --- | --- | --- |
| Image-caption | Web-crawled image-text pairs | Base grounding, wide visual coverage |
| Interleaved docs | Long documents with images | Long-context multimodal understanding |
| Text-only | LLM-quality text data | Language capability, instruction latent |
| Instruction | Curated instruction / VQA / chat data | Task alignment, evaluation-adjacent |

**Training grid.** 1B, 4B, and 8B VLMs, at 6.25B, 25B, 100B, and 200B training tokens. This size × budget grid is what lets DCVLM measure *scaling behavior* of a curation strategy, not just single-point wins.

**Evaluation.** A 52-benchmark suite covering document understanding, chart/plot reasoning, general VQA, OCR, multimodal reasoning, safety, and more. The 33-task core aggregate is the primary score; the wider suite catches capability regressions.

**Headline result.** The DCVLM-Baseline recipe achieves 63.6% on the core suite at 8B / 200B, +5.4pp over FineVision, the previous open-VLM SOTA training corpus. The winning move is not aggressive filtering (which underperforms) but a specific mix that keeps instruction-heavy pools weighted despite their smaller nominal size.

## Why it matters

- **Reframes what "data quality" means for VLMs.** For LLMs the community converged on "filter harder" (Dolma, RefinedWeb). DCVLM's evidence is that for VLMs the marginal return on filtering is small relative to the return on mix composition — a real change in emphasis.
- **Reproducible ceiling for open VLMs.** With a fixed harness and a strong open baseline, new curation ideas can be evaluated without training an entire frontier VLM stack from scratch.
- **Bridges the LLM/VLM data-curation gap.** Text-only tokens play an unexpectedly important role in VLM training; DCVLM makes that first-class instead of an afterthought.
- **Instruction pools scale better than captions.** The scaling curve shows the instruction-heavy mix widening its lead as tokens grow — meaning the "just crawl more captions" playbook actively underperforms at scale.

## Gotchas & tricks

- **Fixed pipeline is a feature and a constraint.** Everything about model / optimizer / LR is frozen. A curation strategy that only wins with a matched training recipe won't show up in DCVLM. Real-world stacks may still benefit from co-designing data and training.
- **Instruction pool overlap with evaluation.** Some instruction-tuning data closely resembles the evaluation benchmarks. DCVLM's decontamination policy matters; strategies that push instruction weight up should be checked against the decontamination baseline.
- **Mixing vs. formatting is entangled.** "Instruction-heavy" mixtures also imply chat-template formatting, which changes token counts and the effective mixture ratio. Report both nominal and post-tokenization mix ratios.
- **The 8B / 200B ceiling is not the frontier.** Closed frontier VLMs train on much more data and much larger models. DCVLM measures *curation quality* at a controlled scale — extrapolation to frontier scale is a research question.

## Sources

- Paper: *DataComp-VLM: Improved Open Datasets for Vision-Language Models* — Farina, Udandarao, Nguyen, et al. (35 authors, LAION / Tübingen / MPI / Stanford / multi-org), 2026 — [arXiv:2606.28551](https://arxiv.org/abs/2606.28551)
- Predecessor: *DataComp: In search of the next generation of multimodal datasets* — Gadre et al., 2023.
