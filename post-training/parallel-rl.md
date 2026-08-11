# Parallel-RL
*Depth — decouple multi-task RL post-training instead of stacking SFT stages.*

**TL;DR:** In multi-task LLM post-training, SFT gradients across tasks interfere destructively while on-policy RL updates are approximately sparse and orthogonal. Parallel-RL exploits this by training multiple task-specific RL policies in parallel (rather than one monolithic multi-stage SFT run) and merging or routing between them, delivering higher efficiency and flexibility on multi-task benchmarks.

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md), [ppo.md](ppo.md)
**Related:** [gradient-interference.md](gradient-interference.md), [_post-training.md](_post-training.md), [rlvr.md](rlvr.md), [fine-tuning/README.md](fine-tuning/README.md)

---

## What it is

Standard multi-task LLM post-training pipes tasks through sequential SFT stages, or mixes them into one large batch. Both variants suffer from **catastrophic interference**: later tasks overwrite earlier ones, and gradient collisions across tasks force painful data-mixture tuning.

Parallel-RL flips the setup. Because RL post-training induces sparse, near-orthogonal weight updates per task (see [gradient-interference.md](gradient-interference.md)), you can train **one RL policy per task independently and in parallel**, then compose them — via merging, routing, or serving multiple heads — without the interference tax that kills the SFT version.

## How it works

1. **Per-task RL runs.** For each task $T_i$ in the multi-task set, launch an independent RL run (GRPO / PPO with the appropriate reward) from the same shared base checkpoint.
2. **Variance-limited updates.** Advantage normalization + on-policy sampling keep each per-task update small in variance, so per-task weight deltas $\Delta\theta_i$ have small mutual overlap.
3. **Composition.** Combine the resulting checkpoints — one of: weight-space merging (e.g. task arithmetic $\theta_{\text{merged}} = \theta_{\text{base}} + \sum_i \Delta\theta_i$), routing (task-conditional expert selection), or serving as separate heads.

The key claim: interference between per-task updates is bounded by *gradient variance* (a small quantity under advantage normalization), not by *gradient norm* (which is what SFT is stuck with). So merged Parallel-RL degrades gracefully as you add tasks.

## Why it matters

- **Sequential multi-stage SFT is fragile.** Every added task risks overwriting earlier capability; data mixtures require painful re-tuning for every new task.
- **Parallel-RL parallelizes across tasks.** Independent RL runs → wall-clock scales with the slowest task, not the sum.
- **New tasks don't force re-training old ones.** Add a task = run one more RL job and re-merge; skip the full multi-stage rebuild.
- **Empirical evidence.** The parent paper shows RL-induced updates are sparse and near-orthogonal at the parameter level, unlike SFT — the theoretical justification for merging without full retraining.

## Gotchas & tricks

- **Base checkpoint must be shared.** All per-task runs start from the same $\theta_{\text{base}}$ or task arithmetic breaks.
- **Merging isn't free.** Simple sum-of-deltas can still degrade if any single task's update is too large. Scale each $\Delta\theta_i$ or apply TIES / DARE-style pruning before summation for many-task regimes.
- **On-policy is load-bearing.** Off-policy RL (or heavily replayed rollouts) can push updates back into the norm-limited regime and re-introduce interference.
- **Doesn't apply to non-verifiable tasks.** For tasks where you'd use SFT anyway (style, refusal), the interference story flips; you still need mixed SFT.

## Sources

- Paper: *SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning Paradigms for LLMs* — Zhu et al., 2026 — arXiv:2608.03573.
