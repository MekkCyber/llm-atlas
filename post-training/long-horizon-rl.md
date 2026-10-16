# Long-Horizon RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Agent RL at 100+-step tool-use horizons needs more than vanilla GRPO: the credit-assignment is long, the gradient singular spectrum is skewed, long rollouts dominate the loss, and off-policy drift explodes. A working recipe: **Muon optimizer**, **token-averaged loss**, **truncated importance sampling**, and **auxiliary length / tool-error penalties**. The combination trains SWE-style issue-resolution agents where AdamW-based baselines stall.

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md), [../pre-training/muon-optimizer.md](../pre-training/muon-optimizer.md)
**Related:** [rl-prompt-curation.md](./rl-prompt-curation.md) · [rejection-sampling.md](./rejection-sampling.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md)

---

## What it is

Most reasoning-RL papers operate at horizons of a handful to a few dozen tokens-worth-of-steps. **Long-horizon agent RL** is the regime of 100+ tool-use steps per rollout: repo-level coding agents, deep research agents, long browser trajectories. Four problems appear only at this scale.

## How it works

Each pathology and the matched fix:

### 1. Skewed gradient spectrum → Muon

A long trajectory's gradient is dominated by a handful of steps where the key decision happened. The gradient tensor's singular spectrum becomes highly non-uniform, and AdamW's element-wise scaling cannot normalize the step direction. [Muon](../pre-training/muon-optimizer.md) orthogonalizes the update's singular values, so no single direction dominates.

### 2. Long rollouts dominate loss → token-averaged loss

Standard RL losses sum over tokens then average over rollouts in a batch. For wildly varying rollout lengths (a 10-token "I failed" vs a 20k-token trace), long rollouts contribute 2000× the per-rollout gradient of short ones. **Token-averaged loss** (sum then divide by total token count, not by rollout count) rebalances contribution per token rather than per rollout:
$$L = \frac{\sum_i \sum_t \ell_{i,t}}{\sum_i |o_i|}.$$

### 3. Off-policy drift → truncated importance sampling

Over a long rollout, the behavior and target policies drift apart. The GRPO ratio $\pi_\theta / \pi_{\text{old}}$ accumulates multiplicatively over hundreds of steps and explodes. **Truncated IS** clips the ratio at a bounded value (above and below), stabilizing the gradient at the cost of some bias.

### 4. Trajectory-shape pathologies → length and tool-error penalties

Even with correct final rewards, rollouts can be absurdly long (filler tokens, retrials) or feature avoidable tool errors. Shaping rewards with:
- a **length penalty** (negative reward per token beyond a soft budget),
- a **tool-error penalty** (negative reward per failed tool call),

keeps the policy from exploring these failure modes while still reaching the correct answer.

Combined with GRPO under rule-based verifiable reward (SWE-style unit tests, patch-apply success), the stack trains stably at horizons where either pure GRPO or AdamW-based GRPO fail.

## Why it matters

- **Named a regime.** "Long-horizon" is not just more steps — the four pathologies above kick in together, and all four need to be addressed. One-fix recipes do not scale.
- **Reproducible on two model families.** The paper reports higher training success than baselines on SWE-style benchmarks across two architectures, suggesting the recipe generalizes.
- **Portable to other 100+-step domains.** Deep research agents, browsing agents, multi-turn tool-use agents all plausibly benefit.

## Gotchas & tricks

- **Length penalty is a double-edged sword.** Too strong and the agent gives up on hard problems; too weak and rollouts bloat. Tune on a validation set of task budgets.
- **IS truncation introduces bias.** Clipping too tight makes the gradient a biased estimate; the paper uses asymmetric clips (looser on the "doing well" side).
- **Muon alone isn't sufficient.** Replacing AdamW without also adopting the token-averaging and IS fixes gives smaller gains or regressions.
- **Tool-error penalty needs careful attribution.** A tool call failing because the environment is flaky is not the policy's fault; blind penalization biases the policy away from recovery.
- **Rollout cost explodes.** 100+-step rollouts under $G$-way GRPO are expensive — pairs well with [KV cache quantization](../inference/kv-cache-quantization.md) and [partial rollouts](../systems/partial-rollouts.md) for throughput.

## Sources

- Paper: *Long-Horizon RL Needs Muon* — Ruan, Lin, Chen, Zhou, Wei, Huang, 2026 — [arXiv:2610.20256](https://arxiv.org/abs/2610.20256). SWE-style issue-resolution benchmarks on two model families.
