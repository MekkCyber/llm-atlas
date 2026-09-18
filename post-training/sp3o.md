# SP³O — Sparse PPO Critic Supervision
*Depth — a fix for the "value flattening" failure mode of PPO critics on LLMs.*

**TL;DR:** In PPO for LLMs, ground-truth state values estimated from multiple Monte Carlo continuations vary sharply across intermediate response tokens, but the trained critic's predictions stay comparatively flat — a failure mode named **Value Flattening**. SP³O (Sparse PPO) supervises the value loss on only a handful of well-separated states per response — three is enough on Qwen3-Base — and the critic learns to track the true value curve. No policy-loss changes.

**Prereqs:** [ppo](ppo.md)
**Related:** [grpo](grpo.md) · [_rl](_rl.md) · [_rewards](_rewards.md) · [rlvr](rlvr.md)

---

## What it is

Standard PPO applies the value-function loss $L^{\text{VF}} = (V_\theta(s_t) - V_t^{\text{target}})^2$ at **every** token in a response. Li et al. (2026) show that this dense supervision, combined with the temporal correlation between adjacent LLM states, produces a critic whose predictions are systematically smoother than the true value curve. Named **Value Flattening** — verified both on Qwen3-Base LLM RL and in a controlled FrozenLake setup where the effect worsens as state-space grows.

Two mechanisms drive it (both derived in the paper):
1. **Implicit variance penalty.** Averaging squared errors over correlated states biases the critic toward the mean.
2. **Redundant gradient signal.** Temporally correlated states share ~identical gradients — each step's effective information is smaller than the token count suggests.

## How it works

Given a rollout of length $T$:
1. Pick $k \ll T$ **well-separated** anchor states (paper uses $k = 3$; separation is temporal, not learned).
2. Apply the standard MSE value loss **only at those anchors**:
$$
L^{\text{VF}}_{\text{SP}^3\!\text{O}} = \frac{1}{k}\sum_{t \in \text{anchors}} \left( V_\theta(s_t) - V_t^{\text{target}} \right)^2
$$
3. Leave the policy loss (clipped surrogate + KL to reference) unchanged.

The change is one line in the training loop: mask the value loss to $k$ indices per response.

## Why it matters

- **Rescues PPO for LLM RL.** Value-free methods (GRPO, mirror descent) took off partly because the critic was "flat and useless" in practice. SP³O says the critic is *over-supervised*, not doomed — a near-free fix restores its usefulness.
- **Reopens the RM-RL vs verifier-RL choice.** For reward-model RLHF where GRPO's return-baseline doesn't fit (variance issues, sparse absolute reward), a working PPO critic matters again.
- **Zero extra compute.** Fewer value-loss backward operations per response, not more.

## Gotchas & tricks

- **"Well-separated" matters.** Anchors clustered near the response end (or all near the start) recover less of the value curve than temporally spread ones.
- **$k$ is small on purpose.** The paper reports $k = 3$ working across model sizes; large $k$ (say $k > 32$) re-approximates dense supervision and the flattening returns.
- **Diagnose via MC agreement.** The paper's diagnostic is to sample multiple MC continuations at intermediate states and compare their spread to critic predictions — an easy plot to reproduce before adopting.
- **Doesn't apply to value-free RL.** GRPO / RLVR have no critic to sparsify. Orthogonal.

## Sources

- Paper: *Rethinking Critic Learning in PPO: Understanding and Mitigating Value Flattening* — Li, Yan, Luo, Wang, Wang, Tan, Tian, Cui, Ding, Zhao, Li, Cheng, CUHK / Tencent / Shanghai AI Lab, 2026, arXiv 2609.18708.
- Paper: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017 — the base algorithm this modifies.
