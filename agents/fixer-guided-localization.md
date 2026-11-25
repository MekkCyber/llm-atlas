# Fixer-Guided Localization
*Depth — training a code localizer by what a frozen fixer can actually fix, not by what a patch edits.*

**TL;DR:** In a two-stage code agent (locate → fix), the files a patch edits are not always the files the fixer most needs to read. Fixer-Guided Localization (FGL) trains the localizer against the *fixer's own likelihood* of producing a verified patch: propose candidate localizations, score each by `log P(verified_patch | localization)` under a frozen fixer, then fit the localizer to those scores with weighted SFT. No tests, no RL.

**Prereqs:** [README](README.md), [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [SFT README](../post-training/fine-tuning/README.md)

---

## What it is

A distillation target for the localization head of a two-stage code agent. The signal is the downstream fixer's likelihood of patching correctly given a candidate localization — i.e., the localizer is trained to point at code that is *useful to the fixer*, not code that is *in the edit*. Introduced in Hennara et al. (2026) for the Kaggle Gemma-4 Developer Agent competition, built on Gemma 4 E4B.

## How it works

Pipeline per training example:

1. Run the localizer, sample $K$ candidate localizations per issue.
2. For each candidate, run the frozen fixer and compute $s_k = \log P(\text{verified-patch} | \text{candidate}_k)$ without executing tests.
3. Convert scores to soft weights (softmax or likelihood-ratio).
4. Fine-tune the localizer with weighted SFT, pushing probability mass toward candidates with high fixer likelihood.

Trained on 1,629 test-verified Kimi K3 trajectories. No test execution and no RL loop are needed — the frozen fixer's token-level probability is the reward.

## Why it matters

Decouples the two training problems in modern code agents: you can train the fixer on verified patches, then train the localizer against the *fixer-you-have*, aligning the two without executing tests during training. The locality reduction (27.5% less code read while matching the teacher's patch rate) is the practical payoff — shorter context means faster inference and more budget for the fixer.

## Gotchas & tricks

- **The frozen fixer must be fluent on the task.** A weak fixer gives noisy likelihoods, and training the localizer against noise just overfits noise.
- **Likelihood-weighting ≠ rejection sampling.** Here every candidate contributes proportionally to its score; rejection sampling throws away all but the max. The soft version empirically trains better on small data.
- Trained agents explore ~3× more than the base model and solve up to 11/78 development tasks vs. 8 for the base. Remaining bottlenecks at 4B under an 8-minute budget: time, repetition loops, edit quality.
- Doesn't require ground-truth edit locations in training data — only verified patches and candidate localizations.

## Sources

- Paper: Let the Fixer Decide: Training a Code Localizer from the Fixer's Likelihood — Hennara et al., 2026 — [arXiv:2611.12164](https://arxiv.org/abs/2611.12164)
