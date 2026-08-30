# CritICL: Weak-Model-Failure Critiques as ICL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An inference-time reasoning method that skips repeated generation and external verifiers. LLM failure modes are **structured across scales within the same family**, so failures produced by a *smaller* sibling model are a cheap proxy for what to warn a *larger* model about. CritICL harvests weak-model failures, converts them into critique-style in-context examples, and prepends them to the strong model's prompt. Two variants: dynamic (retrieval-augmented, per-input) and static (a fixed global failure profile).

**Prereqs:** [long-cot-rl.md](long-cot-rl.md), [../grpo.md](../grpo.md)
**Related:** [../cot-reward-model.md](../cot-reward-model.md), [prm.md](prm.md), [orm.md](orm.md), [../../evaluation/aime.md](../../evaluation/aime.md)

---

## What it is

A test-time scaling approach that trades **generation compute for context compute**. Instead of sampling many candidate solutions and voting, CritICL fills the prompt with critique-format examples — "here is a failure mode we see on inputs like this, and why it's wrong" — mined from a weaker sibling model. The strong model then answers with the failure modes in context and (empirically) sidesteps them without spending extra decodes.

## How it works

**Offline: build a failure-mode bank.**

1. Run a weaker sibling ($M_\text{weak}$) on a labeled training set.
2. For each failed rollout, generate a **critique**: a short structured natural-language artifact that names the mistake pattern and the correction ("the model dropped the negation of the inequality; the correct step is …").
3. Index critiques by input-type features (task family, step-type signatures).

**Inference-time — two variants:**

- **CritICL-dynamic.** For each new input $x$, predict its likely failure modes (a lightweight classifier over input features), retrieve matching critiques from the bank, and inject them as ICL examples in the strong model's prompt.
- **CritICL-static.** Build a single **global failure-mode profile** — a fixed prompt prefix summarizing the top-$k$ recurring failure patterns for the whole task. Cheaper, no retrieval, but less input-specific.

The strong model $M_\text{strong}$ produces its answer in one pass conditioned on the critiques.

## Why it matters

Reframes **weak-to-strong** from a training-time alignment tool into an inference-time efficiency mechanism. Beats standard ICL and matches or beats test-time-scaling baselines (self-consistency, majority vote, best-of-N) at **significantly fewer generations and lower token cost**. Uses the fact that "what smaller siblings get wrong on this input" transfers up-scale better than random hard examples do.

## Gotchas & tricks

- **Same model family matters.** Failure modes transfer up-scale reliably *within* a family (Qwen weak → Qwen strong) but degrade across families — cross-family critiques act more like generic prompt clutter.
- **Critique quality > critique quantity.** More than a handful of critiques crowds context and hurts. Retrieval quality dominates over bank size in the dynamic variant.
- **Static is a strong baseline.** For narrow tasks (competition math), the static failure profile captures most of the gain — try it before building a retrieval pipeline.
- **Not a training method.** No parameter updates; drop-in at inference time.

## Sources

- Paper: *CritICL: Inference-Time Weak-to-Strong Generalization from Small Language Model Failure Modes* — Wu et al., 2026 — [arXiv:2608.27455](https://arxiv.org/abs/2608.27455)
