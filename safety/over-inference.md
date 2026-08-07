# Over-Inference (Persistent-Memory LLMs)

*Depth — a fabrication failure mode of memory-augmented LLMs, and the Self-Monitoring Inversion that breaks cross-model self-audit.*

**TL;DR:** Personalized LLMs with persistent memory (ChatGPT Memory, Claude Projects, Copilot Memory) fabricate user attributes beyond what evidence supports — this is **over-inference (OI)**. Sun, Zhang & Sheng (2026) build MirageBench (150 personas × 6 tasks × 143,616 judged claims × 12 models across 7 families) and find every model over-infers 35–49% of the time. The headline result is a **Self-Monitoring Inversion**: at the model-selection level, models' *self-assessed* OI is *negatively* rank-correlated with their measured OI (ρ=−0.60) — the models that report the least fabrication are flagged as fabricating the most. Within a single model, self-audit still discriminates (AUROC 0.58–0.83).

**Prereqs:** [../evaluation/README.md](../evaluation/README.md)
**Related:** [../agents/memory-staleness.md](../agents/memory-staleness.md) · [cot-monitoring.md](cot-monitoring.md) · [safety-case.md](safety-case.md)

---

## What it is

Persistent-memory LLMs update a user profile across turns and re-use it in later prompts. Over-inference is when the model *invents* attributes the profile doesn't actually support (stated demographic, preference, or history) — a memory-driven analogue of hallucination.

The paper formalizes OI via a four-way faithfulness taxonomy (grounded / partially-grounded / ungrounded / contradicted), operationalized by an LLM judge validated against blind human annotators (Cohen's κ=0.86 four-class, 0.90 binary).

## How it works — the evaluation

- **MirageBench structure.** 150 personas across three types (stereotypical, counter-stereotypical, neutral), 6 personalization tasks spanning an "imagination gradient" (from tightly grounded factual queries to open-ended imagined preferences), 12 models across 7 families, 143,616 judged claims.
- **Independent judge.** An LLM judge scores every claim as one of four faithfulness categories, validated on 400 human-annotated claims.
- **Cross-model correlation.** Compare each model's self-reported OI (ask the model to audit its own claims) against the independent judge's OI score, rank models by each.

## Key finding: Self-Monitoring Inversion

- **Across-model:** ρ = −0.60 (p = 0.044), n = 12 — models that self-report low OI are ranked high-OI by the independent judge. Wide CI [−0.90, +0.06], marked exploratory.
- **Within-model:** the same self-audit still discriminates each model's own good vs. bad claims — AUROC 0.58–0.83. Self-critique correlates with truth *inside* a model but inverts *across* models.

Cross-model mean OI: 41.6% (claim-weighted 41.8%). Task-dependent: 27%–59%. Multi-turn pilot shows inferred attributes accumulate approximately linearly with little revision.

## Why it matters

- **Puts every persistent-memory product on the hook.** No model in the leaderboard escapes 35% over-inference. Memory-driven personalization is currently a fabrication surface, not a solved feature.
- **LLM-as-judge with self-comparison is broken by construction.** The Self-Monitoring Inversion generalizes beyond OI: any evaluation that trusts a model's self-assessment for cross-model ranking inherits the inversion risk.
- **External verification, not self-report.** The paper's normative recommendation — verify memory claims against source turns with an independent judge, don't ask the model whether it fabricated.
- **Multi-turn accumulation is silent.** Attributes pile up without revision — a memory system that never re-evaluates old inferences drifts monotonically toward more fabrication.

## Gotchas & tricks

- **Judge model is not neutral.** MirageBench's independent judge is itself an LLM; results are calibrated against human annotation but a different judge choice may shift absolute OI rates. Rank ordering is more stable than absolute rates.
- **Within-model self-audit still works.** If you're comparing prompts or memory-write policies *for one model*, the model's self-critique is usable. Cross-model comparison is what inverts.
- **Persona balance matters.** Stereotypical personas produce different OI patterns than counter-stereotypical — some models fabricate more when the persona hints at a stereotype, which surfaces bias in a measurable way.
- **Task-dependence 27–59% means the reported average understates the worst cases** on imagination-heavy tasks.

## Sources

- Paper: *The Personalization Mirage: How LLMs Fabricate User Profiles, and Why Self-Monitoring Misleads* — Sun, Zhang, Sheng, 2026 — [arXiv 2608.04570](https://arxiv.org/abs/2608.04570).
