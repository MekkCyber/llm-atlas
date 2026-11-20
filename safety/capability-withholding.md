# Capability Withholding (as a Safety Control)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The practice of **removing training data** for a target skill (a programming language, a scientific domain, a style) to keep a model from learning it. Controlled experiments across chess, Othello, molecular strings, bracket languages, web text, and a programming language withheld from a 7B model show that withholding **delays but does not prevent** skill acquisition: a withholder-trained model recovers the skill with a small fraction of the data a from-scratch model would need. Capability withholding is a speed bump, not a lock.

**Prereqs:** [../data/_data-curation.md](../data/_data-curation.md)
**Related:** [../data/decontamination.md](../data/decontamination.md), [README.md](README.md)

---

## What it is

A proposed safety mechanism for dangerous capabilities (biology, chemistry, cyber): if the training corpus lacks the relevant material, the model shouldn't be able to help with those tasks. This has been assumed to be a durable control.

Marosi & Antal (2026) test the assumption directly with controlled withholding experiments. For each target skill:

1. Pretrain a model with the skill-related data withheld.
2. Measure the model's ability on the skill (near-zero).
3. Give it a small amount of the withheld data via continued pretraining or SFT.
4. Compare the data required to reach a target capability level vs a from-scratch model's requirement.

The withheld model always recovers the skill — with much less data than from scratch.

---

## How it works (as a safety control)

In practice, withholding looks like:

- **Pretraining-data filters** on keywords, domains, URLs, or classifiers for the target capability.
- **Multi-stage pipelines**: filter at crawl time, classifier at curation time, decontamination against known evaluation sets.
- **Supplemental post-training filters** on fine-tuning and instruction data.

Each stage is porous: classifiers miss rephrasings, crawl filters miss transformations, and the model can transfer capability from adjacent non-withheld domains.

---

## Why it matters

- **Quantitative evidence** that withholding does not produce a durable capability block. For any target skill, the recovery data cost is a small fraction of from-scratch cost — and the model has already acquired general cognitive machinery from the non-withheld data that makes the skill cheap to re-learn.
- **Policy implications.** Withholding can slow proliferation (useful as a defense-in-depth layer) but should not be relied upon as the sole safety control for dangerous capabilities. Pair with evaluation-time filters, deployment monitoring, and capability evals.
- **Transfer generality.** The recovery result holds across symbolic (chess, Othello, brackets), domain-specific (molecules), and natural-language domains. Suggests the effect is driven by general-capability transfer, not the specifics of any one skill.

---

## Gotchas & tricks

- **Withholding completeness matters more than it looks.** Residual contamination lets even nominally withheld models score nontrivially. Decontamination-grade filtering is a precondition for interpreting any withholding result.
- **Transfer from adjacent skills.** You can't withhold "chess" and expect the model to be ignorant of all strategic board-game reasoning. The general pattern transfers.
- **Instruction-following amplifies recovery.** A model already instruction-tuned on general data can pick up the withheld skill with even less recovery data than a base model.
- **Capability estimation should bound recovery, not initial state.** A safety case that relies on "the model doesn't know X" should be budgeted against how cheaply X can be re-learned in deployment.

---

## Sources

- Paper: *General Progress Substitutes for Specific Training Data in Language Models* — Marosi, Antal — 2026 — [arXiv:2611.24844](https://arxiv.org/abs/2611.24844).
- Background: *Measuring and Mitigating Training Data Contamination* — various, 2023–2025.
- Related framework: dangerous-capability evaluations (ARA, BioLM, cyber-eval) — measure what withholding is supposed to prevent.
