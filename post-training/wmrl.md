# WMRL — World Model RL

*Depth — replace real environment execution with a learned world model in agent RL, and correct the world model's biased/noisy reward.*

**TL;DR:** RL post-training of research / coding / embodied agents is dominated by *environment* wall-clock: each rollout occupies one real sandbox for real seconds, while generation batches for free. World Model RL (WMRL) swaps the environment for a learned world model that predicts states and rewards, and adds two mitigations — **Online Debiasing** (align the world-model reward against a tiny pool of true rollouts) and **Inverse-Variance Denoising** (down-weight high-variance advantages) — so a *good-enough* world model still converges. 3–4× training speedup; 4B / 9B agents beat 48B / 120B open-weight agents on held-out benchmarks.

**Prereqs:** [_rl](_rl.md), [grpo](grpo.md), [rlvr](rlvr.md)
**Related:** [rl-prompt-curation](rl-prompt-curation.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md), [ppo](ppo.md)

---

## What it is

Agent RL rollouts are asymmetric: the actor's forward pass shares GPU compute across a batch, but each *environment step* (compile, run tests, control a browser, control a robot) is one exclusive process on one machine. As trajectories grow to hundreds of turns, environment execution eclipses generation and becomes the training bottleneck.

WMRL is the class of RL post-training methods that replace the environment with a *learned* transition/reward model — a "world model" — for most of training, keeping true rollouts only as an occasional anchor. The world model is small and batched; execution cost stops scaling with trajectory length.

## How it works

Standard RL update with a verifier-based reward becomes:

```
for step in range(N):
    prompts        = sample_from_dataset()
    responses      = policy.generate(prompts)          # cheap, batched
    if real_step:                                       # rare
        states, r_true = env.run(responses)
    states, r_wm   = world_model(prompts, responses)   # cheap, batched
    r_used         = r_wm  (+ true samples for anchoring)
```

Two corrections stabilize learning against imperfect `r_wm`:

- **Online Debiasing.** Maintain a running estimate of the residual `r_true - r_wm` on the anchor rollouts and add it back as a correction term. Cancels systematic bias in the world-model reward.
- **Inverse-Variance Denoising.** Each rollout's advantage is scaled by the inverse of its estimated reward variance, computed from repeated world-model evaluations. High-variance advantages get down-weighted; the policy update is dominated by the confidently-scored rollouts.

Both are proved to strictly improve the convergence guarantee over naive world-model RL.

## Why it matters

- **Environment cost is the ceiling.** Terminal-Bench-style agents run 300+ tool-call turns; each turn is a real shell. WMRL lifts that ceiling without sacrificing convergence.
- **Small models catch up.** WMRL-post-trained 4B / 9B AutoResearch agents outperform 48B / 120B open-weight baselines on held-out tasks — the training-time savings translate into more updates per unit compute, and the small model closes the gap.
- **Generalizes across domains.** The paper shows transfer to embodied VLA post-training, which shares the "execution is the bottleneck" structure.

## Gotchas & tricks

- **Anchor rollouts are non-optional.** Without a small pool of true rollouts, the debiasing correction is undefined and the policy will drift into world-model artifacts.
- **Variance estimates need multiple world-model queries per rollout.** Trade a few extra world-model forward passes for the denoising signal — the world model is cheap, this is usually fine.
- **World-model errors are worst on OOD rollouts.** A confidently-wrong world-model reward on a novel behavior can lock the policy in. Curriculum toward higher difficulty helps, as does KL-anchoring to the SFT model.
- **Not a replacement for RLVR verifier design.** The world model *predicts* the verifier's output; a well-defined verifier remains a prerequisite.

## Sources

- Paper: *Scaling Automatic Research Agents via World Models* — Yang, Sarwar, Cheng, Shi, Li, Chen, Zhang, Fan, Guo, He, Liao — UIUC, 2026 — [arXiv:2608.12564](https://arxiv.org/abs/2608.12564).
