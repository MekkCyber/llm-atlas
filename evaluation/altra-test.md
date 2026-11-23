# AltRA-Test
*Depth — a statistical test for deciding whether an LLM judge can replace human annotators, without requiring shared annotators across items.*

**TL;DR:** Alt-Test (prior work) asked "can an LLM replace one of the human annotators?" but required every item to be labeled by the same annotator pool — rarely true in practice. **AltRA-Test** drops that assumption: it resamples items and item-level annotator sets, swaps *one* human label per item for the LLM's, and tests whether inter-annotator agreement drops by more than an adaptive margin derived from the observed human-human agreement. Consistent under the paper's assumptions and ~50% pass rate across LLM-dataset combinations over text, image, audio, and video.

**Prereqs:** [README](README.md)
**Related:** *(none yet — this is the first judge-methodology depth file)*

---

## What it is

A hypothesis test with:

- **Null:** replacing one human label per item with the LLM's label does not degrade inter-annotator agreement beyond an adaptive margin.
- **Alternative:** the drop exceeds that margin, so the LLM is not a valid substitute.

Unlike classical inter-rater agreement comparisons, the test does **not require a shared annotator pool**: each item may have been labeled by a different subset of humans, which matches the reality of most crowdsourced or expert datasets.

## How it works

Given an annotation dataset where each item $i$ has a set $A_i$ of human labels (possibly from different annotators across items):

1. **Resample.** Draw bootstrap samples of items and of per-item annotator labels.
2. **Substitute.** For each resample, in each item, pick one human label at random and swap it for the LLM's label on that item.
3. **Compute agreement.** Measure inter-annotator agreement (Fleiss-$\kappa$, Krippendorff-$\alpha$, or task-appropriate) on the resampled set.
4. **Adaptive margin.** Derive the acceptable drop in agreement from the *observed human-human agreement variance* — a more variable human baseline tolerates a larger drop before the LLM is rejected.
5. **Decide.** If the drop exceeds the adaptive margin with statistical significance, reject the LLM as a substitute for this dataset.

Simulations show AltRA-Test outperforms existing baselines in statistical power and type-I control when the shared-annotator assumption is violated.

## Why it matters

- **Matches real datasets.** Most annotation corpora don't have shared annotator pools. Alt-Test's restriction ruled out most practical use; AltRA-Test lifts it.
- **Reality check on LLM-as-judge.** Across text/image/audio/video tests, ~50% of LLM-dataset combinations pass. "LLM judges can replace humans" is far from universally true.
- **A principled gate.** Instead of ad hoc agreement correlations, provides a statistical decision procedure with known properties.
- **Portable to new modalities.** The construction doesn't assume text-specific structure; the authors exercise it across modalities.

## Gotchas & tricks

- **The adaptive margin is the sensitive knob.** Tighter margins reject more LLMs; looser margins accept more. The paper's calibration is reasonable but worth re-tuning per domain.
- **Single-label substitution.** The test substitutes *one* human per item. For tasks with few annotators per item this is a large relative perturbation; interpret pass/fail in that context.
- **Agreement metric choice matters.** Different metrics have different sensitivity to the swap; Fleiss and Krippendorff disagree on borderline cases.
- **Passing AltRA-Test is necessary, not sufficient.** It says the LLM is comparable *in agreement*, not that it reproduces the same decisions in important edge cases.

## Sources

- Paper: *AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators* — Omar, Dror, Gal, Reichart, Calderon, 2026 (Technion, Haifa, Google Research).
- Paper: *Alt-Test* — Calderon et al. (predecessor test requiring shared annotators).
