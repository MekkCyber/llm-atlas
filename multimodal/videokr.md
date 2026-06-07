# VideoKR
*Depth — a knowledge- and reasoning-intensive training corpus and benchmark for video-language models.*

**TL;DR:** Most video-LM corpora are describe-the-frame data — useful for grounding, useless for reasoning. **VideoKR** (2026) is a 315K-example training corpus over 145K CC-licensed expert-domain videos, designed to push VLMs from "say what you see" to "reason over what is happening." It pairs with **VideoKR-Eval**, a benchmark for knowledge-heavy temporal questions where the answer requires domain expertise plus multi-frame reasoning, not just frame description. Models post-trained on VideoKR outperform prior approaches on knowledge-intensive video reasoning.

**Prereqs:** [multimodal/README.md](./README.md), [data/_data-curation.md](../data/_data-curation.md)
**Related:** [evaluation/README.md](../evaluation/README.md), [data/quality-filtering.md](../data/quality-filtering.md)

---

## What it is

Two artefacts:

1. **VideoKR (training corpus).** 145K videos across professional and expert domains (e.g. surgery, sports tactics, engineering, scientific demonstrations), with 315K reasoning examples. Each example is a question that requires domain knowledge plus multi-frame temporal reasoning, not just frame-level description.
2. **VideoKR-Eval (benchmark).** A held-out evaluation set sharing the corpus's task distribution. Probes knowledge + reasoning that ordinary video-QA datasets don't.

Both are CC-licensed and built for post-training — the corpus is large enough to be a primary training signal, not just an eval probe.

## How it works

### Human-in-the-loop generation pipeline

The corpus is built by a pipeline that combines automatic candidate generation with targeted human verification:

1. **Domain selection.** Choose expert-content domains where surface-level description is insufficient (you cannot answer "is this surgery going well" by describing pixels).
2. **Video sourcing.** CC-licensed expert content with sufficient temporal complexity.
3. **Candidate question generation.** A VLM proposes question-answer pairs that require knowledge + reasoning. Templates target multi-step inference, not single-frame description.
4. **Human verification.** Experts validate that the answer is correct and that solving the question genuinely requires knowledge beyond visual description.
5. **Difficulty calibration.** Questions are stratified by reasoning depth; the corpus covers a range from "simple expert lookup" to "multi-step causal inference."

### Post-training recipe

VLMs post-trained on VideoKR see substantial gains on VideoKR-Eval and on prior knowledge-heavy benchmarks. The gain is data-driven — the same architectures improve simply by seeing this distribution.

## Why it matters

- **Targets the video-LM data bottleneck.** Architectural advances in video models have plateaued in part because the training distribution is mostly web-scraped descriptive content. Expert-domain reasoning data is the closest analogue to what made text-LM reasoning take off (math, code).
- **CC-licensed.** Many large-scale video corpora are murky or proprietary. VideoKR being CC-licensed makes it usable in open releases.
- **Domain coverage.** Expert domains tend to have implicit causal structure that frame-by-frame VLMs don't pick up. Sustained training on them seems to nudge architectures toward genuinely temporal reasoning.

## Gotchas & tricks

- **Verification is the cap on quality.** Human verification at this scale is expensive; the paper trades some breadth for verified depth.
- **Domain shift between corpus and eval is a known risk.** If the eval is too close to the training distribution, gains overstate transfer. The held-out construction mitigates but doesn't eliminate this.
- **Not a replacement for general video-LM data.** VideoKR is a *reasoning* signal; you still need broad description-style data for grounding. Mix it in.
- **The "knowledge" axis is implicit.** What counts as "expert knowledge" depends on the domain. The corpus errs on the side of "would a non-expert misanswer," which is roughly the right bar.

## Sources

- Paper: *VideoKR: Towards Knowledge- and Reasoning-Intensive Video Understanding* — 2026 — [arXiv:2606.05259](https://arxiv.org/abs/2606.05259).
