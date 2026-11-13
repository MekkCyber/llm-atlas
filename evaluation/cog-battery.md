# CogBattery
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark that evaluates language models on *text analogues of 13 classical cognitive-psychology paradigms* (n-back, Stroop, task switching, confidence calibration, gambling tasks, etc.) and compares them to published human reference data *at the condition level*, not just the mean-accuracy level. Tested on 54 open-weight models from 23 families. Key finding: models often match human **effect directions** but behave very differently inside the task, and larger models are not reliably more human-like.

**Prereqs:** [README.md](README.md)
**Related:** [../interpretability/README.md](../interpretability/README.md)

---

## What it is

Cognitive psychology evaluates humans not on "overall accuracy" but on *condition contrasts*: the difference between Stroop-congruent and Stroop-incongruent response times, the elevated responding on n-back lures vs non-targets, the slope of confidence-vs-accuracy curves. CogBattery translates 13 such paradigms into text form and compares models against published human curves per condition.

The result is a signed effect-size vector per model, not a scalar score.

## How it works

1. **Paradigm translation.** For each cognitive task, write a text interface that preserves the condition structure (target vs lure, congruent vs incongruent, etc.) and gives the model one answer per trial.
2. **Human reference curves.** Pull per-condition means and error bars from published human studies.
3. **Condition-level scoring.** For each model × task × condition, compute the effect size and compare against the human distribution. A model that matches *directions* but inflates non-target false alarms is distinguished from one that matches both.
4. **Confidence protocol.** Collect both verbal confidence ("how sure are you?") and token-probability-derived confidence; score each separately against human confidence calibration.

## Why it matters

- Reframes "model acts human-like" from an overall-accuracy claim to a condition-by-condition claim. A model matching human mean score while inverting the key effect is no longer "human-like".
- Finds that **larger models are not consistently more human-like**, which is a sharp counterweight to scale-only extrapolation on behavioral benchmarks.
- Gives interpretability work a condition-level target: when a 70B model matches a human effect direction but fails a confidence calibration, you know *which* mechanism to look at.

## Gotchas & tricks

- Text-translated cognitive tasks are not psychometrically equivalent to the original — treat CogBattery scores as a model-vs-model behavioral index, not a claim about the original cognitive construct.
- Verbal vs token-probability confidence diverge on most models; always report both.
- Predefined task families (executive function, memory, reasoning) do *not* predict performance better than random groupings of the same size — don't aggregate by family.
- Many models exhibit a single "always same wager" strategy; filter those before confidence analysis.
- Human reference data is drawn from published studies on English-speaking adult samples; comparisons to models trained in other languages are not calibrated.

## Sources

- Paper: *CogBattery: Human-Referenced Evaluation of Cognitive Task Behavior and Structure in Language Models* — Jiang, Lin, Yamada (2026) — [arXiv:2607.24999](https://arxiv.org/abs/2607.24999)
