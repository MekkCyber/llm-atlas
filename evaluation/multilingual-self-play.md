# Multilingual Self-Play
*Depth — evaluating cross-lingual skill differences by pitting two copies of the same model against each other in different languages.*

**TL;DR:** Two instances of the same model play a text-based game against each other, one per language. Because model, opponent, rules, state space, and available actions are all identical, any performance gap between the two copies is *caused by language*. This isolates cross-lingual **skill** inconsistency orthogonally from knowledge, translation quality, and benchmark-selection artifacts.

**Prereqs:** [mmlu.md](mmlu.md)
**Related:** [../safety/low-resource-language-jailbreak.md](../safety/low-resource-language-jailbreak.md)

---

## What it is

Standard multilingual evaluation of LLMs is dominated by benchmark translations — MMLU-in-language-X, GSM-in-language-X — which conflate skill differences with translation artifacts, knowledge coverage, and dataset contamination in the source language. Cheng et al. (2026) propose a cleaner design: **hold everything except language constant, then measure the outcome gap.**

## How it works

- **Setup.** Two instances of the *same model* play a text-based game against each other, each interacting through a different language interface (game rules, board state, available actions, chat, all presented in that language).
- **What's constant.** Model, opponent (also this model), rules, state space, available actions.
- **What varies.** The natural-language interface only.
- **Metric.** Win–loss margin between the two language copies; also invalid-action rate and strategic tendencies.

Implemented as a multilingual extension to **TextArena**. Evaluated on three open-weight models across **eight languages** and **six games** covering spatial reasoning, imperfect information, resource allocation, and repeated interaction.

**Findings.**

- The same model exhibits markedly different playing strength across languages — systematic variation in win–loss margins, invalid actions, and strategic tendencies.
- Detailed analyses reveal language-specific failures in spatial reasoning, card-conditioned decisions, and optimal move selection.
- **Reasoning-language swap.** In several settings, changing *only the intermediate reasoning language* (while keeping the interface language fixed) recovers much of the lost performance — evidence that language affects different stages of the decision process, not just the surface.

## Why it matters

"Multilingual LLM" claims have been benchmarked with confounded evals. Self-play strips out translation quality, knowledge coverage, and dataset contamination in one move by holding everything except language constant. It also opens a per-stage lens on where the language effect enters (interface vs reasoning language), which is directly actionable for training: languages that lose ability in the reasoning stage can be helped by allowing internal reasoning in a stronger language.

## Gotchas & tricks

- **Only measures skills expressible as games.** Spatial reasoning, imperfect information, resource allocation, repeated interaction — but not open-ended writing quality or factual recall. Self-play is a *skill* eval, not a general-purpose multilingual replacement.
- **Same model, same opponent = symmetry-breaking is required.** Games must have asymmetric rewards or non-trivial state; symmetric zero-sum toys converge to trivial equilibria.
- **Bilingual reasoning breaks internal-language uniformity.** If the "intermediate reasoning language" swap recovers performance, the model isn't really operating in the interface language internally — a nuance worth reporting.

## Sources

- Paper: *Skill Issue: Are Skills Language-Invariant in LLMs?* — Cheng et al., 2026 — [arXiv:2608.25832](https://arxiv.org/abs/2608.25832)
