# Evol-Instruct
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A synthetic-instruction expansion technique that mutates a seed instruction into harder variants using a small catalog of "evolution operators" (deepen, add constraint, increase reasoning depth, change format, etc.). Each operator is a prompt template applied by a strong LLM. Used originally in WizardLM (2023) and extended in VeriEvol (2026) with *type-aware* operators conditioned on problem type for multimodal-math reasoning data.

**Prereqs:** [_data-curation.md](_data-curation.md)
**Related:** [../post-training/_post-training.md](../post-training/_post-training.md) · [falsification-verifier.md](falsification-verifier.md) · [agentic-data-curation.md](agentic-data-curation.md)

---

## What it is

Generating instruction-tuning data from scratch is expensive and biased toward the generator's defaults. Evol-Instruct starts from a small seed set of human-written instructions and iteratively *mutates* each into harder variants by prompting a strong LLM with one of a handful of operators. After a few rounds, the pool covers a wider difficulty and structure distribution than the seeds alone.

## How it works

Two families of operators run in alternation:

| Family | Operators | Effect |
| --- | --- | --- |
| In-depth | Deepen (more reasoning steps), Add constraint, Increase reasoning, Concretize | Harder versions of the same task type |
| In-breadth | Replace topic, Generate new instruction in different domain | New task types around the seed |

Each round: pick a seed (or current-pool item), pick an operator, prompt a strong LLM with `<operator template + seed>`, accept the new instruction, generate a response, optionally filter. Repeat to scale.

**Type-aware extension (VeriEvol).** Plain Evol-Instruct mutates blindly; type-aware Evol-Instruct first classifies the seed (algebra vs. geometry vs. counting vs. …) and applies only operators known to produce *meaningfully harder* variants for that type. This avoids degenerate mutations (e.g., "add a constraint" applied to a counting problem often just adds noise).

## Why it matters

- Cheap difficulty scaling: a 10K human-seeded set can be expanded to 250K with one round of mutations, and the resulting model is much better than one trained on the seeds alone.
- The operator catalog is the surface where domain knowledge enters — controllable, auditable, swap operators per domain.
- Combined with a verifier (cf. [falsification-verifier.md](falsification-verifier.md)) the generation→verification loop becomes a self-sufficient synthetic-data factory for verifiable domains.

## Gotchas & tricks

- The mutated instructions are only as good as the generating LLM's understanding of the operator. Weak operators degrade quickly across rounds.
- Without an answer verifier, errors accumulate — the strong LLM hallucinates harder problems whose "correct" answers are wrong. Pair with a verifier in any serious pipeline.
- Operator distribution shifts the resulting model. Heavy "add constraint" yields longer, more pedantic outputs; heavy "concretize" yields narrower domains.
- Distinct from rejection sampling: Evol-Instruct expands *prompts*; rejection sampling filters *responses*. Often combined.

## Sources

- Paper: *WizardLM: Empowering Large Language Models to Follow Complex Instructions* — Xu et al., 2023 — original Evol-Instruct.
- Paper: *VeriEvol: Scaling Multimodal Mathematical Reasoning via Verifiable Evol-Instruct* — Zheng et al., Tsinghua / Tencent Hunyuan, 2026 — [arXiv:2606.23543](https://arxiv.org/abs/2606.23543) — type-aware operators + falsification verifier.
