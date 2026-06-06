# Memorization & Extraction
*Depth — evaluating whether LLMs reproduce training data verbatim, distinguishing what they can be forced to leak from what they actually leak in practice.*

**TL;DR:** Memorization eval has two regimes: **capability** ("can the model be coerced into outputting training data via adversarial prefixes?") and **propensity** ("does the model produce training data under ordinary, non-adversarial prompts?"). The two diverge sharply: most modern LLMs are vulnerable to prefix-completion extraction but rarely leak verbatim under normal use. **PropMe (2026)** is the framework that contrasts the two regimes systematically; the gap matters for legal, policy, and threat-modeling claims.

**Prereqs:** [data/deduplication](../data/deduplication.md)
**Related:** [data/decontamination](../data/decontamination.md)

---

## What it is

Memorization in LLMs = training-data sequences that can be recovered (verbatim or near-verbatim) from the trained model. Evaluations differ in how aggressively they probe:

- **Prefix-completion attacks.** Take a known training string, feed the first half as prompt, measure whether the model continues with the second half. This is a *capability* test — it asks whether the model *can* be made to leak.
- **Discoverable memorization.** Search the output space (via random prefixes, fuzzing) for any verbatim training match.
- **Membership inference.** Given a string, predict whether it was in the training set.
- **Propensity eval (PropMe).** Use *non-adversarial* prompts that resemble normal user queries; measure rate of verbatim training-data emission in that distribution.

Capability ≠ propensity. A model that completes prefixed training strings 30% of the time might emit verbatim training data <0.1% of the time in normal conversation.

## How it works

PropMe-style propensity evaluation:

1. **Identify candidate strings** — known training sequences (or strong dedup-detection candidates).
2. **Design natural prompts** — prompts a real user might issue around the topic of the string, not adversarial prefixes.
3. **Sample many completions** per prompt.
4. **Report two numbers**:
   - Capability rate: adversarial-prefix extraction rate.
   - Propensity rate: verbatim emission rate under natural prompts.
5. **Disagreement is the signal.** Per-model, per-string, per-domain.

The framework is method-agnostic about the attacks — what matters is reporting both axes side-by-side.

## Why it matters

- **Legal & policy framing.** Copyright, privacy, and GDPR claims often turn on "does the model reproduce X in deployment?" — that's propensity, not capability. Reporting only worst-case extraction overstates real exposure.
- **Threat model alignment.** A red team coercing the model with prefix attacks is a different adversary from a normal user. Defensive measures should be evaluated against the right threat.
- **Mitigation diagnosis.** Dedup, decontamination, and refusal training each shift the two numbers differently. A mitigation that drops capability while leaving propensity unchanged tells you something different from one that drops both.

## Gotchas & tricks

- **Verbatim ≠ memorized in the privacy-relevant sense.** Near-verbatim, paraphrased, or fact-level leakage is harder to measure but often the actual concern. Pair string-match metrics with semantic similarity.
- **Natural-prompt design is the hard part.** "Natural" smuggles judgment calls — what does a real user ask? PropMe formalizes a propensity-aware *contrast* but doesn't standardize the prompt distribution.
- **Watch base vs. instruct.** Instruction-tuned models often have much lower propensity than their base counterparts (refusal training filters), while capability under adversarial prefixes can be similar.
- **Dedup is the primary upstream fix.** Most measured memorization tracks training-data duplication count. See [deduplication](../data/deduplication.md).
- **Decontamination is for eval, not privacy.** [Decontamination](../data/decontamination.md) prevents benchmark leakage, which is a *different* memorization question (benchmark-set vs. broad-web).

## Sources

- Paper: *LLMs Can Leak Training Data But Do They Want To? A Propensity-Aware Evaluation of Memorization in LLMs* (PropMe) — Barmina, Schneider-Kamp, Galke, 2026 — [arXiv:2606.06286](https://arxiv.org/abs/2606.06286) — primary source for the propensity-aware framing.
- Paper: *Extracting Training Data from Large Language Models* — Carlini et al., 2021 — foundational extraction-attack paper, capability-side.
- Paper: *Quantifying Memorization Across Neural Language Models* — Carlini et al., 2023 — relates memorization to model scale and data duplication.
