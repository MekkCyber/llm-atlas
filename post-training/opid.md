# OPID — On-Policy Skill Distillation
*Depth — extract dense token-level supervision for agentic RL from the policy's own on-policy successes.*

**TL;DR:** Outcome-based RL for language agents gives only a single trajectory-level reward — a stable optimization backbone, but useless for telling *which* intermediate decision deserves credit. Existing skill-conditioned variants try to bridge the gap with external skill memories or retrieved privileged context, but the retrieved skills are usually drawn from a distribution that mismatches the *current* policy's state distribution. OPID sidesteps both problems by extracting hierarchical skill supervision directly from the agent's own completed on-policy trajectories — by construction, the supervision distribution matches the rollout distribution.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md), [reasoning/prm.md](reasoning/prm.md)
**Related:** [rejection-sampling.md](rejection-sampling.md), [rl-prompt-curation.md](rl-prompt-curation.md), [progress-advantage.md](progress-advantage.md)

---

## What it is

A training framework that interleaves outcome-based RL (GRPO/PPO-style) with **token-level self-distillation** on the policy's own successful rollouts. The distillation signal is *always on-policy* — it's the policy distilling its own wins — so the train/test state-distribution gap that plagues retrieved-skill methods disappears.

## How it works

For each RL training step:

1. **Rollout.** Sample $K$ trajectories per prompt with $\pi_\theta$, score with the outcome reward.
2. **Skill extraction.** From the *successful* trajectories in the batch, cluster the steps into hierarchical skills — coarse skills (e.g., "search-then-answer") at the top, finer skills (e.g., "issue follow-up query") at the bottom. Clustering is done on hidden states or action tokens; the hierarchy is built without external labels.
3. **Skill-conditioned token-level distillation.** Construct an SFT-style loss on the successful trajectories where each step is conditioned on its inferred skill. This loss is interleaved with the GRPO update.
4. **Policy update.** Combine the GRPO objective with the skill-distillation loss; backprop both.

Two things make this distinct from prior skill-conditioned RL:

- The skill memory **is the current batch**, not a static external store.
- No retrieval at inference — the policy has already internalized the skill conditioning during training.

## Why it matters

- **Credit assignment without PRMs.** Reaches dense token-level supervision without training a separate process reward model — a major win for agent settings where PRM construction is intractable.
- **No skill-library maintenance.** External skill memories rot as the policy improves; OPID's "skills" auto-refresh every rollout.
- **Composable.** Drops into any outcome-reward agent RL stack (GRPO, PPO, RLVR for agents).
- Consistent gains across ALFWorld, WebShop, and Search-based QA, with the largest improvements on long-horizon tasks where outcome reward is sparsest.

## Gotchas & tricks

- Requires enough successful rollouts per batch to make clustering stable — early in training, when success rates are low, you may need to boost $K$ or pretrain with SFT.
- The hierarchical skill structure is the load-bearing piece; flat clustering loses much of the advantage. Hyperparameters for hierarchy depth depend on task length.
- Doesn't change the outcome reward — won't fix a fundamentally misaligned verifier. Use alongside good reward design, not as a substitute.

## Sources

- Paper: *OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning* — Yang, Wu, Lu, Shen, Zhang, Feng, Zhang, Luo, Lian, Wen, Tao, 2026 — [arXiv:2606.26790](https://arxiv.org/abs/2606.26790). Tsinghua / Zhejiang / CUHK / NTU / Tongji.
