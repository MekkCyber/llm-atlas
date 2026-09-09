# Boundary-Aware Refusal Training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A self-distilled pipeline that trains a model to refuse **narrow, deployment-specific boundaries within a topic** rather than blanket topics. The same base model can be tuned to refuse political persuasion for one deployment while still discussing politics analytically for another. Built from four moves: controlled topic generation, coverage repair by escalating retries, in-distribution compensation data, and harmful-benign pairs — all self-generated, no external labels.

**Prereqs:** [../post-training/fine-tuning/README](../post-training/fine-tuning/README.md), [safety/README](./README.md)
**Related:** [../post-training/rejection-sampling](../post-training/rejection-sampling.md), [mismatched-generalization](./mismatched-generalization.md), [refusal-suppression](./refusal-suppression.md)

---

## What it is

"Safety" for a hosted model is not one thing — a legal-drafting assistant needs a different refusal boundary from a debate coach even inside the identical topic of "political persuasion." Blanket-refusal training conflates these deployments and either over-refuses (blocks legitimate use) or under-refuses (leaks unsafe outputs). Boundary-aware refusal training explicitly targets the *narrow boundary within a topic* as the object of learning.

The recipe is offline and self-generated:

1. **Controlled topic generation** — enumerate a controlled space of prompts on the target topic that straddles the boundary.
2. **Coverage repair** — iteratively escalate the retry budget until every intended-refusal prompt has an accepted refusal trace. In the paper: prompts lacking accepted refusal traces dropped from 19.88% to 0.20%.
3. **In-distribution compensation data** — for each refused case, generate a benign-side analogue on the same topic so the model does not learn to refuse the *topic* itself.
4. **Harmful-benign pairs** — explicit pair supervision that pins the boundary between refused and allowed content.

## How it works

Training is standard SFT (or DPO-style) on the resulting mixed dataset. The novelty is entirely in the data pipeline; the loss is off-the-shelf. Evaluation checks *both* sides of the boundary — under-refusal on the harmful side, over-refusal on the benign side (XSTest-style).

## Why it matters

- **Trains a controllable dial rather than a global switch.** Different products can be tuned to different points on the same topic.
- **Reported numbers on Qwen3-8B (political persuasion target):**
  - Target-domain refusal: **9.47% → 84.75%**.
  - Unsafe responses across benchmarks: **26.26% → 0.14%**.
  - XSTest over-refusal: rises to **74.00%** — the safety-usability tradeoff is *explicit and measurable*, not hidden.
- **Removes the human-label bottleneck.** All refusal traces are self-generated with retry escalation.

## Gotchas & tricks

- **Over-refusal is the price.** The XSTest jump is the honest evaluation of the tradeoff; deployments that need low over-refusal must dial back the retry-escalation coverage target or compensate with more benign-side data.
- **The "boundary" is defined by the seed prompts.** Bias in the topic-generation stage propagates directly into what the trained model refuses.
- **Not a jailbreak defense.** Boundary-aware refusal is orthogonal to adversarial robustness — the pipeline shapes the *intended* boundary, not the model's behavior under attack.

## Sources

- Paper: *Safety for Whom? Boundary-Aware Self-Distillation for Controlled LLM Safety Refusal* — López-Ávila, García-Ferrero, Garcia, Tiene, Orús, 2026 — [arXiv:2609.04482](https://arxiv.org/abs/2609.04482).
