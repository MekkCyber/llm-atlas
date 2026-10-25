# Reasoning-token provenance
*Depth — tracing the origin of self-correction tokens ("wait") across pretraining / mid-training / SFT / RL.*

**TL;DR:** The characteristic self-correction token "wait" in R1-style reasoning models is almost never *invented* by reinforcement learning. It is inherited from training data (mid-training corpora, SFT reasoning traces) and then amplified or reshaped by RL. Where the token doesn't appear in data, RL rarely produces it on its own. The paper traces the per-stage provenance of revision tokens and finds the data-side signal, not the RL loss, is doing the work.

**Prereqs:** [long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [mid-training.md](../pre-training/mid-training.md)
**Related:** [grpo.md](../post-training/grpo.md), [cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

A methodology for attributing specific token behaviors in a trained model to specific pipeline stages. The paper applies it to "wait", "but", "actually", and other revision markers in long-CoT models and separates three questions: did the token *exist* at this stage; did its *frequency* change at this stage; did its *downstream behavior* (what follows "wait") change at this stage?

## How it works

1. **Checkpoint along the pipeline.** Keep base-pretraining, mid-training, SFT, and RL-stage checkpoints.
2. **Measure per-stage statistics.** For each revision token: generation probability on fixed prompts, context under which it appears, mean "revision depth" (how long the correction continues) after it.
3. **Attribute.** The *first* stage at which the token appears with non-trivial probability is the stage that introduced it. Later stages are credited for amplification or reshaping, not creation.
4. **Judge audit.** The paper also runs an LLM judge over revision labels and compares to human annotation — on this task the LLM judge *reversed* human trends, meaning the broader long-CoT literature should sanity-check its judges.

## Why it matters

- **Punctures the "RL emerges reasoning" narrative.** The emergent-behavior claim relies on tokens appearing to come from nowhere; this paper shows they came from upstream data.
- **Actionable:** curating revision-rich mid-training data is probably a larger lever than tuning RL for self-correction. If your mid-training corpus has no "wait", your RL won't put it there.
- **Judge warning.** LLM judges can flip human-labeled trends on revision analysis; the entire self-correction evaluation stack needs re-checking.
- **Transfers.** The provenance methodology generalizes to any token-level behavior claim (reflection, back-tracking, verification).

## Gotchas & tricks

- "Doesn't produce" is softer than "can't produce": the paper measures RL *rarely* invents these tokens, not that it's impossible under some other recipe.
- Token-level provenance can miss *behavioral* emergence that doesn't surface in word choice. "Wait" being inherited doesn't mean the surrounding reasoning was.
- Judge calibration against humans must be done per task; a judge accurate on math grading isn't automatically accurate on revision-behavior labeling.

## Sources

- Paper: *Inherited, Not Invented: Tracing the Provenance of Wait in Reasoning Models* — Kou et al., 2026 — [arXiv:2610.30882](https://arxiv.org/abs/2610.30882)
