# Reasoning-Marker Provenance

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A label-free interpretability procedure for tracing where the *reflection markers* ("wait", "but", "let me check") in a reasoning model's output actually come from. The headline finding on the "wait" token: **RL does not teach it**. Zero-RL on bases that lack the token almost never produce it, even when accuracy rises substantially. The marker arrives through mid-training or SFT on reasoning traces, and RL only amplifies, reshapes, or re-tunes its sensitivity to real errors.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../pre-training/mid-training.md](../pre-training/mid-training.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

Reasoning models (R1-style, o1-style) emit tokens like "wait", "actually", "let me reconsider" that mark places where the model backtracks or re-checks a step. The dominant interpretation in 2025 was that RL *induced* this behavior. The provenance paper tests that claim with four non-label-based measures applied across open models' public training histories (base → mid-training → SFT → RL).

Measurement axes:

1. **Frequency** — how often the marker appears per generation length.
2. **Position** — where in the trace the marker lands.
3. **Form** — surface variations ("wait", "wait,", "wait…") and context.
4. **Usage** — distribution over surrounding content (correction vs filler).
5. **Error-injection test** — inject a known-wrong step into a trace and measure the marker's rate of appearance immediately after.

---

## How it works

### Trace through training stages

For each open model with a public pipeline, run the five measurements at every stage checkpoint: the base LM, after mid-training on reasoning-flavored data, after reasoning-SFT, after RL. Compare distributions stage-to-stage.

### Zero-RL control

Take a base LM that **does not** produce the marker. Apply the same RL recipe used to train the public reasoning model. Measure (a) whether the marker emerges and (b) whether accuracy improves. Finding: accuracy improves substantially; marker almost never emerges from scratch.

### Injected-error sensitivity

Construct test prompts where a step is deliberately wrong. Measure the model's probability of emitting a reflection marker *at* the error. This distinguishes an *error-sensitive* marker (fires at mistakes) from a *filler* marker (fires uniformly). RL moves markers from filler toward error-sensitive; it does not create them.

### LLM-judge check

An LLM judge is used to label "revision" events in generated traces. The paper shows the judge can reverse human-labelled trends — i.e. relying on an LLM to score reflective behavior produces conclusions that disagree with human labels.

---

## Why it matters

- **Corrects a major narrative.** The "RL induced reflection" story drove a lot of R1-era interpretation and research. Relocating the origin to mid-training / SFT data changes what one tries to improve about reasoning models.
- **Mid-training data is load-bearing.** If the marker comes from training-data distribution, data curation (not RL recipe tuning) is where to invest for improving reflection quality.
- **CoT monitoring is weaker than hoped.** If reflection tokens are inherited rather than tied to actual error-checking, CoT monitors trained to look for them will have false confidence in traces that produce the token for stylistic reasons.
- **LLM-judge audits are suspect.** The paper's finding that an LLM judge can reverse human-labeled revision trends is a general warning for using LLM-based evaluators on fine-grained reasoning behaviors.

---

## Gotchas & tricks

- **"Wait" is one marker.** The provenance analysis is anchored on this one token; other markers ("but", "actually", "hmm") may have different provenance. The methodology generalizes but the specific claims are about "wait".
- **Open-models-only study.** Closed-source reasoning models can't be traced through their training stages. The paper's conclusions are only validated on open models whose checkpoints are published.
- **Doesn't argue RL is useless.** RL still improves accuracy a lot — the paper's claim is only that RL isn't the *source* of the reflection vocabulary. Attribution of accuracy gains to RL is unchallenged.
- **Error-sensitivity is a weaker property than reasoning.** Even an error-sensitive marker is pattern-matching, not metacognition. The paper is careful not to claim that the shift to error-sensitive usage implies real self-correction capability.

---

## Sources

- Paper: *Inherited, Not Invented: Tracing the Provenance of* Wait *in Reasoning Models* — Kou, Chen, Yang, Luo, Peng, Zhang, Han, Guo, HKUST / Huawei Noah's Ark Lab / Peking University, 2026 — [arXiv:2610.30882](https://arxiv.org/abs/2610.30882).
- Related: *DeepSeek-R1* technical report — the R1-style training recipe this paper audits ([../case-studies/deepseek-r1.md](../case-studies/deepseek-r1.md)).
