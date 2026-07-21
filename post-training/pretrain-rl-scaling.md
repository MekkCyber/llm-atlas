# Pretraining × RL Compute Scaling
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reasoning-oriented RL post-training is usually studied with pretraining held fixed. In practice, **how much RL compute buys you depends on the pretraining checkpoint you start from**. Shen et al. (2026) run a controlled sweep on a chess testbed (where move quality is checkable) and derive **joint scaling laws** coupling pretraining size / data to RL post-training returns — plus a characterization of what RL actually modifies inside the network.

**Prereqs:** [_rl.md](_rl.md), [rlvr.md](rlvr.md)
**Related:** [grpo.md](grpo.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [../pre-training/README.md](../pre-training/README.md)

---

## What it is

Reasoning-RL literature since R1 treats pretraining as fixed context: "given a base model, RL adds X." But two obvious questions have been open:

1. **How do pretraining choices (model size, data mixture) reshape the returns to RL compute?** — i.e. is there a scaling law that jointly parameterizes both axes?
2. **What does RL actually do to the model?** — beyond aggregate benchmark bumps, what is measurably changing inside the network?

The paper answers both in a controlled setting: chess as the target task (ground-truth move quality is cheap via Stockfish), a sweep over pretraining model sizes × pretraining data × RL compute, and both scaling-law fitting and inside-the-network probing.

## How it works

- **Chess as the testbed.** Cheap ground truth (per-move quality vs. Stockfish), verifiable rewards for RL, natural for LLM tokenization (PGN / algebraic notation). Chess also isolates reasoning from world knowledge — a rare property.
- **Sweep pretraining × RL.** Pretraining sizes and data mixtures on one axis; RL compute budget on the other. Each cell of the grid produces a checkpoint scored on the same held-out chess evaluation.
- **Fit joint scaling laws.** Look for parametric families of the form $L(\text{pretrain}, \text{RL}) = L_\infty + a \cdot P^{-\alpha} + b \cdot R^{-\beta} + \text{interaction}$ — the interaction term is the interesting one.
- **Probe the network before / after RL.** Compare hidden-state / logit-level statistics on the same prompts to characterize what RL modifies rather than what it adds.

## Why it matters

- **Which base is worth spending RL on?** The joint law tells you when RL returns diminish for a given base — informing the "buy more pretraining or spend more RL" decision that every reasoning-model team faces.
- **Rejects the "RL is free" default.** RL returns are not independent of pretraining; the same RL budget on a weaker base can waste compute the base wouldn't have wasted on more tokens.
- **Provides a mechanism story.** Rather than a black-box benchmark lift, the paper reports what RL is actually doing inside the model — feeding into interpretability and diagnosis of RL failures.
- **Portable methodology.** Chess is a testbed of convenience; the sweep-and-fit recipe generalizes to any RL-target domain with cheap ground truth (code, math).

## Gotchas & tricks

- **Chess is one domain.** The scaling-law coefficients likely don't transfer numerically to math / code / open-ended reasoning; the *shape* of the law probably does.
- **Single-seed sweeps are noisy.** For the numbers to be trustworthy, expect multiple seeds per grid cell — expensive but non-negotiable.
- **Pretraining-data mixture matters as much as size.** A big base trained on the wrong mixture may under-perform a small base trained on the right one, and the paper explicitly disentangles these.
- **Watch for reward-model / verifier confounds.** In chess, Stockfish is the verifier; in domains without a Stockfish, the RL signal is a proxy and the "scaling law" is really a proxy-scaling law.
- **Interaction term is the interesting coefficient.** A joint fit that reports only the marginals is telling you the same story as running each axis alone.

## Sources

- Paper: *Understanding Reasoning from Pretraining to Post-Training* — Shen, Li, Rahman, Sun, Goldblum, Telgarsky, Izmailov — NYU / Modal / UCLA / UIUC / Columbia, 2026 — [arXiv:2607.16097](https://arxiv.org/abs/2607.16097).
- Related: *Training Compute-Optimal Large Language Models* (Chinchilla) — Hoffmann et al., 2022 — the canonical pretraining scaling-law paper.
- Related: *DeepSeek-R1* — DeepSeek, 2025 — the canonical RL-on-base recipe the paper is quantifying.
