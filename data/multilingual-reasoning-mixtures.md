# Multilingual Reasoning Data Mixtures
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Frontier reasoning models default to thinking in English even under non-English prompts. The fix is *not* per-language fine-tuning; it's **carefully composed data mixtures** during reasoning-style SFT. Sweeping per-language shares of reasoning traces produces compact recipes that transfer L2 reasoning across typologically diverse languages while preserving English capability. Cohere's Tiny Aya L2-Thinker (3.35B) reasons in the user's language at **>93% rate across 60 languages** on 6 benchmarks.

**Prereqs:** [_data-curation.md](./_data-curation.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [quality-filtering.md](./quality-filtering.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

"L2 reasoning" is the behavior of *producing the internal chain-of-thought in the user's language*, not just the final answer. Standard reasoning fine-tunes are dominated by English CoT traces, and the resulting models silently code-switch: they reason in English then translate the final response, which introduces drift on culturally-loaded, math-notation-sensitive, and instruction-following tasks.

The paper's central claim: **reasoning is a language-agnostic substrate** once instilled — so the training-side lever is not a new algorithm, it's the *mixture*.

## How it works

Setup:
- Fix a base model (Tiny Aya, 3.35B).
- Vary the **per-language share** of reasoning-style SFT data (math, commonsense, instruction-following, generation, cultural reasoning).
- Include a small "language identifier" signal so the model can condition on the target reasoning language.

Sweep dimensions studied:
- **Number of languages represented** in the mix (low → typologically diverse coverage).
- **Share ratio** — English-dominant vs. balanced vs. long-tail-emphasized.
- **Task diversity per language** — math-only vs. spanning IF/generation/cultural.

Findings (paraphrased):
- Small shares of many languages transfer *reasoning* more effectively than a large share of a few.
- Reasoning quality per language decouples from raw pretraining coverage — a language with modest pretraining exposure still benefits strongly if it appears in the reasoning mixture.
- English capability is preserved under balanced mixtures; catastrophic forgetting requires much more aggressive shifts.

## Why it matters

- **Removes a stubborn UX bug in frontier reasoners.** "Reason in English, respond in Chinese" behavior has been visible in Kimi, DeepSeek-R1, and Qwen reasoning specialists; this paper makes it a *data engineering* problem with a compact solution.
- **Cultural evaluation actually improves.** Because CoT now happens in-language, cultural-reasoning benchmarks that were degraded by English intermediate steps recover.
- **Cheap to reproduce.** No new architecture, no new RL algorithm — just a curated SFT mixture. The recipe transfers to any reasoning-capable base.

## Gotchas & tricks

- **Cold-start matters.** The base model needs enough per-language *pretraining* coverage for the mixture-based recipe to bootstrap; a model that has essentially no exposure to a language won't magically start reasoning in it from a small SFT share.
- **Long-tail languages need explicit inclusion.** The paper reports meaningful gains for languages that would otherwise sit at <1% share; balanced mixtures over-favor high-resource languages if not deliberately corrected.
- **Task diversity per language is the second-order lever.** A language covered only by math traces produces a strong math-reasoner-in-that-language but weak IF or cultural performance in-language.
- **Language-consistency reward is a complement.** DeepSeek-R1 uses an in-training language-consistency reward at RL time; this paper achieves similar behavior at SFT time via the mixture. Either can be layered.
- **Evaluation must test in-language reasoning explicitly.** Standard multilingual benchmarks score the final answer, not the CoT language. The paper introduces the "L2 reasoning rate" as a first-class metric.

## Sources

- Paper: *Building Multilingual Bridges: Data Mixing as the Pillar of Generalization for In-Language Reasoning* — Mofakhami, Sahu, Salamanca, D'souza, Bérard, Euyang, Fadaee, Kreutzer (Cohere Labs), 2026 — arXiv:2609.10445.
