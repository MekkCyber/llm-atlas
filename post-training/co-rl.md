# Co-RL (Cohort RL)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Co-RL trains a **cohort of decoupled models** (no shared parameters) simultaneously with RL, where each model's reward comes from its **peers' agreement/disagreement** rather than a ground-truth verifier. Reasoning ability emerges from cohort diversity alone. Matches or beats supervised RLVR on text and multimodal reasoning benchmarks — with zero labels.

**Prereqs:** [_rl.md](./_rl.md), [ppo.md](./ppo.md), [grpo.md](./grpo.md), [rlvr.md](./rlvr.md)
**Related:** [rl-prompt-curation.md](./rl-prompt-curation.md)

---

## What it is

An **unsupervised** RL post-training recipe. Where RLVR requires a per-prompt verifier (`is_correct: str → bool`), Co-RL derives the reward from the **cohort itself**: several independently-parameterized models are trained together, and each one's reward is a function of how the other models score its outputs.

Diversity is a first-class ingredient — if the cohort collapses into agreement, the reward signal vanishes.

## How it works

1. **Cohort.** Instantiate $K$ policies $\{\pi_1, \dots, \pi_K\}$ with disjoint parameters (different base checkpoints, seeds, or architectures).
2. **Rollout.** For each prompt, each $\pi_k$ produces its own answer $y_k$.
3. **Peer reward.** For each $y_k$, the reward $r_k$ is a function of how the *other* $K-1$ models rate or reproduce $y_k$ — for example, the fraction of peers that vote $y_k$ correct via a critique prompt, or the average likelihood peers assign to $y_k$.
4. **RL update.** Each $\pi_k$ is updated with a standard policy-gradient loss (paper uses a GRPO-style group-relative advantage inside its own rollouts) against $r_k$.
5. **Diversity term.** An explicit or implicit penalty keeps cohort outputs from collapsing (e.g. a KL term to the *initial* checkpoint of each model, so different seeds diverge along different axes).

## Why it matters

- **Label-free RL.** Removes the verifier bottleneck — the biggest reason RLVR is confined to math/code today.
- **Matches supervised methods.** +3.0–8.6% on text reasoning and +2.3–7.2% on multimodal reasoning over base models; comparable to or better than label-based RLVR baselines the paper compares against.
- **Multimodal transfer.** The recipe works across modalities because the peer-reward signal doesn't care what task the models are solving, only that they can rate each other.

## Gotchas & tricks

- **Cohort diversity is the reward.** If you initialize all $K$ models from the same checkpoint with the same seed, you get zero signal — everyone agrees on everything, right or wrong. Diverse seeds/checkpoints matter more than large $K$.
- **Susceptible to shared bias.** If all $K$ models were pretrained on similar data, they can be *consistently wrong* in the same way, and Co-RL will reinforce the shared bias. The paper mitigates with mixed base families.
- **Compute footprint is $K\times$ single-model RL.** All $K$ models train simultaneously; there is no shortcut to sharing the base weights (that would kill diversity).
- **Not verifier-free forever.** Papers using Co-RL still evaluate against ground truth on held-out tasks — the training loop is label-free, not the eval.
- **Reward hacking mode: cohort collusion.** If two models start echoing each other's outputs verbatim, they receive high mutual reward for no reason. Adding a diversity/novelty penalty is essentially required.

## Sources

- Paper: *Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL* — Yang et al., UCSD / JHU, 2026 — [arXiv 2608.17253](https://arxiv.org/abs/2608.17253) — introduces the peer-reward cohort RL recipe.
