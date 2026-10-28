# ProAS — progress-aware advantage shaping for multi-turn agentic RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Critic-free RL (GRPO and its variants) is cheap but gives weak token-level credit on long, multi-turn agent trajectories. ProAS is a plug-in on top of GRPO that improves credit assignment **without adding a critic**: it builds a per-batch **dynamic reference graph** of trajectory steps, masks unreasonable incentives, and asymmetrically redistributes penalties. On Qwen2.5-1.5B/7B it reports **91.9% / 94.3%** peak success on ALFWorld and **66.7% / 70.8%** on WebShop, with up to **2.44×** faster convergence.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [rlvr.md](./rlvr.md), [rl-prompt-curation.md](./rl-prompt-curation.md), [ppo.md](./ppo.md)

---

## What it is

A modification to GRPO's advantage computation for the multi-turn setting. In vanilla GRPO, the group-relative advantage is a single scalar per rollout, broadcast to every token. For a 50-step agent trajectory, this gives one bit of credit for the whole run. ProAS replaces the broadcast with a graph-structured redistribution: steps that look reference-redundant across the batch get their advantage flattened, and steps that are genuinely load-bearing get a stronger, asymmetrically shaped signal.

Crucially, the reference graph is **extracted from the batch of rollouts itself** — no learned critic, no extra model.

## How it works

Per batch:

1. **Build the reference graph.** For each prompt, cluster the $G$ rollouts by progress structure — which intermediate states they visit, which tool calls they share. Edges connect steps that occur in a reference-redundant pattern across clusters.
2. **Identify load-bearing steps.** Steps that distinguish successful from unsuccessful rollouts within the group are tagged load-bearing. Steps that look identical across success/failure (reference-redundant) are tagged flat.
3. **Shape advantages.** Flat steps get their per-token advantage dampened; load-bearing steps get an asymmetric boost — penalties on load-bearing failures are redistributed more aggressively than rewards on load-bearing successes (to counter sparse positive signal).
4. **PPO-clipped update.** Standard GRPO objective, now with the shaped per-token advantages.

The shaping is parameter-free beyond a small number of mixing coefficients.

## Why it matters

Multi-turn agentic RL has been splitting between "give up and live with GRPO's noisy credit assignment" and "add a critic back and pay the training cost." ProAS is a third path: shape advantages from reference-graph structure visible in the rollouts themselves. Peak scores and convergence gains on ALFWorld/WebShop are consistent with the credit-assignment story — small-model agents (1.5B) see the biggest convergence speedups because they have the least pre-trained structure.

## Gotchas & tricks

- **Needs enough rollouts per prompt for the reference graph.** With small $G$ the clustering is noisy.
- **Benefit shrinks as the critic alternative gets cheaper.** For settings where a critic is feasible (shorter trajectories, small models), S3AC-style actor-critic may match or beat ProAS.
- **Shaping coefficients matter.** The asymmetric penalty boost is a tunable knob; the paper reports defaults but it is not fully hyperparameter-free.

## Sources

- Paper: *ProAS: A Progress-Aware Advantage Shaping Framework for Multi-Turn Agentic Reinforcement Learning* — Ding et al., LINs Lab, 2026 — [arXiv:2610.18683](https://arxiv.org/abs/2610.18683).
