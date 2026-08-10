# Turn-level credit assignment (AgentOPSD)
*Depth — recursive log-odds belief update as a turn-level advantage estimator for multi-turn agentic RL.*

**TL;DR:** In multi-turn tool-use RL the reward is trajectory-level, but only a few pivotal turns actually decided the outcome. AgentOPSD replaces the flat trajectory advantage with a **turn-level advantage** derived from token-level teacher–student log-prob gaps, aggregated per turn and rolled forward as a Bayesian belief in log-odds space. Critic-free, no extra rollouts, drops into GRPO.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [../agents/README.md](../agents/README.md)

---

## What it is

A per-turn advantage estimator for agentic RL. Given a multi-turn trajectory `(u_1, a_1, o_1, u_2, a_2, o_2, …)` and a single terminal reward, AgentOPSD computes a scalar credit per *turn* (not per token, not per trajectory), so the policy update at each turn sees a signal proportional to that turn's causal contribution to the outcome. Positioned as an alternative to (a) trajectory-broadcast advantages (vanilla GRPO), (b) per-token learned value functions (PPO), and (c) learned process reward models (PRM).

## How it works

Three stages per trajectory:

1. **Token-level evidence.** For each response token `t`, compute the log-prob gap between a *privileged teacher* (self-distilled from a stronger-context rollout) and the current policy: `Δ_t = log π_teacher(o_t | ...) − log π_θ(o_t | ...)`. Sum `Δ_t` inside each turn to get turn-level evidence `E_k` for turn `k`.
2. **Recursive belief update.** Maintain a Bayesian belief that "this trajectory will succeed," parameterized in log-odds space: `L_k = L_{k-1} + E_k`. This is exactly the log-odds form of Bayes' rule — evidence adds linearly.
3. **Turn advantage = marginal belief revision.** `A_k = L_k − L_{k-1}` (equivalently `E_k` up to normalization). Feed `A_k` into the standard GRPO objective per turn. Pivotal turns show up as large marginal revisions; filler turns get near-zero updates.

The teacher is refreshed periodically (recursive self-distillation) — the current policy's best-of-N context becomes the next round's teacher.

## Why it matters

Multi-turn agentic RL is bottlenecked by *where* the reward lands, not *what* it is. Broadcasting the terminal reward across every turn muddies the gradient with contributions from turns that didn't matter, and learned per-step scorers (PRMs, learned value nets) need step-level labels or double the compute. AgentOPSD extracts denser supervision from log-probs the rollout already computed — free structure. Reported 89.1% success on ALFWorld with Qwen2.5-7B, beating GRPO and self-distillation baselines.

## Gotchas & tricks

- Requires a *privileged* teacher (self-distilled from higher-context or longer-rollout traces) — with the same-context teacher, evidence collapses to zero.
- Belief in log-odds space keeps updates in a numerically stable range; naive probability-space Bayes underflows on long trajectories.
- Turn boundaries must be defined by the harness (tool call / observation / next action) — subword-boundary "turns" defeat the aggregation.
- Complementary to PRM: log-prob evidence is cheap and always available; PRM is more informative but needs labels. Consider composing them for hard reasoning tasks.

## Sources

- Paper: *AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning* — Wang et al., 2026 — [arXiv:2608.05987](https://arxiv.org/abs/2608.05987)
- Code: https://github.com/ZethWang/AgentOPSD
