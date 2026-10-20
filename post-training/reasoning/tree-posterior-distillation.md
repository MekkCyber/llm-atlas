# Tree-Aware Posterior Distillation (TreeOPSD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A critic-free RL framework for long-horizon multi-turn agents. Rollouts are structured as a **tree** that branches alternative continuations off shared prefixes; terminal rewards are propagated back into **turn-level credit**; that credit is converted into a bounded **teacher distribution**; the policy is trained to match the teacher with a **Forward-KL** objective. The split lets credit assignment live in reward space (via trees) and the policy update live in distillation space, which keeps updates stable across sequential mini-batches.

**Prereqs:** [grpo.md](../grpo.md), [_rl.md](../_rl.md) (policy gradients, KL divergence)
**Related:** [prm.md](prm.md) · [rejection-sampling.md](../rejection-sampling.md) · [long-cot-rl.md](long-cot-rl.md) · [_rewards.md](../_rewards.md)

---

## What it is

Agentic RL with sparse terminal rewards leaves many turns in a long trajectory unlabeled. GRPO handles this by normalizing per-trajectory reward against a group mean, but every turn in a trajectory shares the same scalar advantage. TreeOPSD attacks the credit-assignment problem directly: it rolls out a **tree** of alternative continuations from shared prefixes, assigns credit to each turn based on the outcomes of its descendants, then turns that credit into a target distribution the student policy is distilled toward.

## How it works

**Rollout tree.** For each prompt, sample a trajectory and periodically fork alternative continuations from past turns. Branches share prefix compute (one KV cache per shared prefix), so the extra cost is sublinear in the number of leaves.

**Turn-level credit.** Propagate each leaf's terminal reward back along its path, giving every turn a credit value tied to the outcomes of its descendants.

**Bounded teacher distribution.** Convert per-turn credit into a target distribution $q(a \mid s)$ over the actions actually sampled at that turn, with a cap that prevents a single branch from dominating.

**Forward-KL update.** Train the policy with $\min \mathrm{KL}(q \,\|\, \pi_\theta)$ (student matches teacher's mass). Because the teacher is bounded and does not change within a mini-batch, the update is more stable than a direct policy gradient against the same tree-derived advantages — especially across sequential batches drawn from the same tree.

## Why it matters

- **Decouples credit assignment from policy update.** Any mechanism that produces turn-level credit (tree rollouts, PRMs, learned reward decomposers) can feed the same Forward-KL update.
- **Critic-free, like GRPO.** No value network to maintain; the tree gives a baseline for free via sibling branches.
- **Deep Research gains.** Reported wins over GRPO are largest on long-horizon Deep-Research tasks where the reward signal is sparse and far from the decision point.

## Gotchas & tricks

- **Prefix caching matters.** Without it, trees of width $k$ cost $k\times$ a single trajectory; with shared KV reuse, the overhead is a small multiplier.
- **Teacher bounding is load-bearing.** Unbounded softmax teachers reproduce the direct policy-gradient instability the paper is trying to avoid.
- **Not a replacement for process reward models.** When step-level labels are available (verifier at each turn), PRMs give sharper credit; TreeOPSD trades labels for structure.

## Sources

- Paper: *Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning* — Wang et al., 2026 — [arXiv:2610.23181](https://arxiv.org/abs/2610.23181)
