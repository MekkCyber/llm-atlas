# Context-Length Curriculum
*Depth — pretraining schedule that expands the maximum sequence length in stages across the training run, so the model learns to use each new long-range span rather than treating extra tokens as padding.*

**TL;DR:** Rather than pretraining at the final target context length from the first step, a context-length curriculum trains the model at shorter lengths early (cheap, high throughput) and **steps the maximum length up in stages**, retuning RoPE frequencies (and any window-size hyperparameters) per stage. The pattern converts the "long-context capability" question from an *inference-time extrapolation* problem into a *pretraining-time exposure* problem — the model is trained on real long sequences at each stage rather than asked to extrapolate. Made explicit as a first-class training-recipe component in the ZGCM-1 tech report (2026), which uses it to reach 256K context in a 7B base.

**Prereqs:** [rope](../fundamentals/rope.md), [wsd-schedule](wsd-schedule.md)
**Related:** [mid-training](mid-training.md), [_training-stability](_training-stability.md)

---

## What it is

Long-context capability can be reached in three broad ways:

1. **Train at the target length throughout.** Correct but expensive: early training is compute-inefficient at long sequences (throughput drops, cache dominates).
2. **Train at short length, then extrapolate.** Position-interpolation (PI), YaRN, ALiBi, NTK-aware RoPE scaling. Cheap but fragile; extrapolation quality varies.
3. **Train at short length, then a small mid-training stage at long length.** The most common current practice: most of the run at 4K–8K, then a shorter stage at 32K–128K.

A **context-length curriculum** generalizes option 3 into a full staged schedule. The maximum sequence length is stepped up across the run (short → medium → long), and each stage's data mix, positional-encoding hyperparameters, and any window/attention configuration are tuned for that stage's target length. The model sees real long sequences at each stage; length extrapolation is not asked of it.

## How it works

A typical curriculum has this shape:

- **Stage 1 (short).** Sequences at 4K–8K. The bulk of tokens are burned here at high throughput. RoPE frequency configuration is the default for the base.
- **Stage 2 (medium).** Sequences at 16K–32K. Throughput drops moderately; the model begins seeing genuinely long dependencies. RoPE base frequency (θ) is often re-scaled per the target length. Data mix is biased toward long documents so the model sees actual long-range structure.
- **Stage 3 (long).** Sequences at 128K–256K+. Small fraction of total tokens (long sequences are expensive) but represents the target regime. Attention configuration (sliding-window size, full-attention layer frequency) is tuned for this stage; RoPE base frequency is re-scaled again if needed.

Between stages the model is not re-initialized; training continues from the previous stage's optimizer state. Learning-rate schedule is typically extended or annealed slightly at each transition. See [wsd-schedule](wsd-schedule.md) for a compatible outer LR pattern.

## Why it matters

- **Cheap and capable.** Most of the compute is spent at short length where throughput is highest; long context is reached without paying the long-sequence throughput cost across the whole run.
- **Better long-context quality than extrapolation.** The model is trained on real long sequences at each stage rather than asked to extrapolate at inference, so long-context tasks (long-document QA, needle-in-haystack, tool-mediated agent traces) reflect learned capability rather than extrapolation quality.
- **Composes with architecture.** Bases that use sliding-window attention or MLA still benefit — the curriculum controls what the model *sees*, the architecture controls what the model *pays for*.

## Gotchas & tricks

- **RoPE re-scaling per stage is not optional.** If the RoPE base frequency (θ) stays at the small-context value, long-context training will silently underperform because the positional signal saturates at long distances. NTK-aware scaling per stage is the current default.
- **Data mix must contain real long documents at each stage.** A stage 3 curriculum that trains on stage 1's short-document mix at 256K sequence length is training on padding; the long-document mix is where the capability actually comes from.
- **Optimizer state carries over.** Reinitializing between stages loses the pretraining. Warmup at each stage transition is typically small and short — long warmup wastes tokens.
- **Evaluate long-context capability *at every stage*.** Regressions between stages (e.g. long-context quality drops after transitioning to a new stage) are usually a hyperparameter mismatch (θ, window size, LR) and are cheaper to catch immediately than at the end.
- **Length-generalization is not the same target.** A context-length curriculum trains up to some target length; it does not by itself buy generalization past that length. Extrapolation techniques (position-interpolation, YaRN) still matter for going beyond the curriculum's terminal stage.

## Sources

- Paper: *ZGCM-1: A Fully Open and Extremely Efficient Foundation Model for Math and Agentic Search* — He et al. — 2026 — [arXiv:2609.13356](https://arxiv.org/abs/2609.13356) — uses a progressive-context curriculum to reach 256K context in a 7B base.
- Background: *YaRN: Efficient Context Window Extension of Large Language Models* — Peng et al. — 2023 — the length-extrapolation counterpart.
- Background: *Longformer* — Beltagy et al. — 2020 — the sliding-window architecture typically paired with a length curriculum.
