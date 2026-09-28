# Alignment-Failure Detection with RLCD (Jev)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use a single **RLCD-trained** (Reinforcement Learning for Calibrated Decisions) probe model, **Jev**, to answer many typed questions about one input with calibrated probabilities — in a single forward pass. Zero-shot generic-question prompting hits median AUROC 0.886 across ten alignment failure classes and 44 benchmarks, matching human-agreement rates at 63× lower cost than LLM-judge scorers.

**Prereqs:** [_attacks.md](_attacks.md), [../post-training/_rewards.md](../post-training/_rewards.md), [cot-monitoring.md](cot-monitoring.md)
**Related:** [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Screening deployed LLMs for alignment failures — sycophancy, jailbreaks, deception, prompt injection, hallucination, privacy violation, bias, reward hacking, uncertainty concealment, power seeking — needs a scorer that is **cheap**, **calibrated**, and **general**. LLM-judge scoring is expensive and often uncalibrated; per-failure classifiers (e.g. Llama Guard) score one label per call. Jev, trained with RLCD, can answer multiple typed questions about one input with calibrated probabilities in a single call.

The paper's contribution is turning Jev into a **detection recipe** for alignment failures and quantifying the cost-vs-accuracy tradeoff.

## How it works

Given an input record (prompt + response, optional metadata):

1. **Question conditioning.** Frame each of the ten failure modes as a typed question ("Did the model comply with a jailbreak attempt in this response?").
2. **Field conditioning.** Vary independently which fields of the input Jev sees (prompt only, response only, both, both plus reference, …). This is where the paper's real signal lives: **question wording matters little, context matters more**, particularly fields that encode the label.
3. **Zero-shot classification.** Jev returns a calibrated probability. Threshold or use directly.
4. **Benchmark integration.** RLCDAlignBench packages 44 benchmarks across the ten failures and five target models, letting you score any deployed model at low cost.

## Why it matters

- 63× cost reduction vs. LLM-judge scorers, at comparable agreement with human labels.
- Because Jev is calibrated across many typed questions, the same probe covers a broad menu of failure modes without training a family of classifiers.
- Surfaces **label defects in existing benchmarks** as a side effect — Jev's calibrated disagreements localize errors in the benchmark scorers themselves, an unusual and useful property.

## Gotchas & tricks

- Some failure modes are **relational**: they depend on a reference (user belief, injected instruction) not present in the response alone. Field conditioning is how you handle this — vary what Jev *sees*, not what it's *asked*.
- Generic questions are a strong baseline; over-tuning question wording gives little gain.
- Not a substitute for red-teaming: detection scores existing outputs but does not surface novel attacks. Pair with attack generation.
- RLCD training itself needs a calibrated-decision dataset; adopting the recipe from scratch is non-trivial.

## Sources

- Paper: *Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures* — Guo, Liu, Deng, Li, Zhao, Wu, Chen, Zhang, Zhang — 2026 — [arXiv:2609.29429](https://arxiv.org/abs/2609.29429).
- Code + data: https://github.com/sumleo/RLCDAlignBench
