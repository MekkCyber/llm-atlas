# Memorization Capability vs Propensity
*Depth — audit training-data leakage by both worst-case extractability and ordinary-use leakage rate.*

**TL;DR:** Most "memorization audits" report **capability**: can an adversarial prefix make the model regurgitate a training datum? PropMe (Barmina et al., 2026) argues this is the wrong single number. The same model that *can* leak under crafted prefixes usually *does not* leak in normal use. **Capability and propensity** should be reported separately. Continual pretraining on different data measurably reduces older memorized content — supporting the propensity axis as a real, malleable property.

**Prereqs:** [_attacks](_attacks.md)
**Related:** [../data/decontamination.md](../data/decontamination.md), [../data/deduplication.md](../data/deduplication.md)

---

## What it is

A framework for evaluating training-data leakage in LLMs that distinguishes two regimes:

- **Capability** — given the most adversarial prompt you can construct, can you elicit the training datum? An upper bound on leakage. Captured by attacks like *prefix completion* or *targeted extraction*.
- **Propensity** — under typical user prompts, how often does the model emit the training datum unprompted? An average-case measure.

Auditing one without the other systematically mis-characterizes the safety posture.

## How it works

PropMe's audit recipe:

1. **Capability probe.** Construct adversarial prefixes targeting the training datum (e.g. first $k$ tokens of the document, or a paraphrased lead-in known to precede it). Generate completions; check for exact or near-exact match.
2. **Propensity probe.** Sample a large pool of *non-adversarial* prompts from a topic-matched distribution. Generate completions; measure the rate at which the training datum appears.
3. **Report both numbers separately.** PropMe also runs SimpleTrace, an attribution tool that points back from a generated string to the training-corpus document(s) it most likely came from — useful for confirming an apparent leak is actually a leak.

Empirical finding: across two open models, two datasets, two languages, capability is high (extraction works under crafted prefixes) while propensity is low (ordinary prompts rarely surface training data). Continual pretraining on a different corpus further suppresses older memorization on the propensity axis — capability decays more slowly, propensity decays measurably.

## Why it matters

- **Avoids a recurring policy mis-call.** Reporting only capability triggers "the model leaks!" alarms even when real-world leakage is vanishingly rare. Reporting only propensity hides worst-case risk under adversarial prompting.
- **Matches deployment threat models.** A consumer chatbot threat model is propensity-shaped (random users); an adversarial-extraction threat model is capability-shaped (motivated attackers). Pick the relevant axis per deployment.
- **Gives a knob.** Propensity is reducible by continual pretraining on new data; capability is reducible by training-data choices and decontamination. Different mitigations for different axes.

## Gotchas & tricks

- **Define "match" carefully.** Exact substring match underestimates leakage; loose semantic similarity overestimates it. PropMe uses normalized edit-distance bands.
- **Propensity is prompt-distribution-dependent.** "Ordinary use" varies across deployments. Compute propensity on prompts that represent your actual user mix, not generic web text.
- **Continual pretraining isn't a free fix.** It reduces older-data propensity but adds new memorization risk for the new training data.
- **Combine with deduplication audits.** Duplicated training data is the single strongest correlate of both capability *and* propensity — fix that first.
- **Attribution (SimpleTrace) avoids false alarms.** A "leak" that doesn't actually trace to the training corpus is a hallucinated coincidence, not a memorization event.

## Sources

- Paper: *LLMs Can Leak Training Data But Do They Want To? A Propensity-Aware Evaluation of Memorization in LLMs* — Barmina, Schneider-Kamp, Galke, U. Southern Denmark, 2026 — [arXiv:2606.06286](https://arxiv.org/abs/2606.06286) — introduces PropMe and SimpleTrace, names the capability-vs-propensity split.
