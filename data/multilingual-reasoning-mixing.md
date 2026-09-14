# Multilingual Reasoning via Data Mixing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reasoning LMs default to reasoning in English regardless of the prompt language. A carefully composed **SFT data mix** — English reasoning traces + broad multilingual non-reasoning data, scheduled properly — makes a 3.35B model reason *in* the user's prompt language ("L2 reasoning") on **>93%** of prompts across **60 languages**, without collecting reasoning traces in every target language.

**Prereqs:** [_data-curation](_data-curation.md), [../post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md)
**Related:** [README](README.md)

---

## What it is

"L1 reasoning" is the current default: the model reads the prompt in language X, thinks in English, answers in X. It's inaccessible for non-English users, loses cultural nuance, and forgoes knowledge best expressed in X. "L2 reasoning" is the alternative — reason *and* answer in the prompt language.

Tiny Aya L2-Thinker shows L2 reasoning is a **data-composition** problem, not a supervision problem. You don't need reasoning traces in Swahili or Tamil to make the model reason in Swahili or Tamil. You need broad multilingual coverage from cheap non-reasoning corpora, a strong English reasoning backbone, and the right mix.

## How it works

- **Multilingual non-reasoning corpus.** Broad coverage across 60+ languages, from readily available multilingual instruction / chat data. Cheap, no reasoning traces required.
- **English reasoning backbone.** A sufficient volume of high-quality English CoT / long-form reasoning traces to instill *how to reason at all*.
- **SFT data mix + schedule.** The two are blended in an SFT recipe; the paper investigates ratios and orderings. Both matter — pure blend under-transfers; naive stage-then-blend loses coverage.
- **No per-language reasoning traces.** Held-out languages (never seen with reasoning supervision) still reason in-language after training.
- **Backbone.** Applied at 3.35B parameters to produce Tiny Aya L2-Thinker.

## Why it matters

Frontier reasoning is *practically* English-only despite multilingual base models — because reasoning-trace collection scales poorly across languages. If broad multilingual non-reasoning data plus English reasoning is enough to induce L2 reasoning, the field has a plausible path to in-language reasoning for the long tail of underserved languages without linear-in-languages annotation cost.

Reported: >93% L2 reasoning rate on 60 languages across 6 benchmarks (math, commonsense, IFEval-style, open-ended generation, cultural reasoning) while keeping benchmark strength.

## Gotchas & tricks

- **Coverage breadth beats depth per language.** Adding a lot of one extra language did less than adding a little of many.
- **English reasoning cannot be replaced yet.** Dropping the English reasoning backbone tanks L2 reasoning — the paper reads this as reasoning being a language-agnostic *behavior* that must first exist somewhere.
- **Ratio, not sequence.** Blending throughout SFT worked better than staging (multilingual then reasoning, or vice versa).
- **Benchmarks matter.** Reasoning-rate alone is misleading — a model that reasons in-language but wrongly is worse than one that reasons in English correctly. Evaluate on task benchmarks *and* L2-rate.
- **Cultural reasoning is the hardest axis.** Math and commonsense L2-transfer easily; cultural reasoning needs explicit multilingual coverage of culturally-grounded prompts.

## Sources

- Paper: *Building Multilingual Bridges: Data Mixing as the Pillar of Generalization for In-Language Reasoning* — Mofakhami, Sahu, Salamanca, et al. (Cohere Labs), 2026 — https://arxiv.org/abs/2609.10445
