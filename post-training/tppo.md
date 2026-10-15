# Tracking-Value PPO (TPPO)
*Depth — a PPO variant whose critic predicts per-token *changes* in log-odds-of-success, trained from terminal outcomes alone.*

**TL;DR:** Vanilla PPO regresses a value head `V(s_t)` onto Monte-Carlo returns at every token, which is noisy for long reasoning trajectories. **TPPO** reparametrizes the critic: instead of predicting the state value directly, it predicts the *token-wise contribution* to the log-odds of eventual success. These contributions sum into the value logit, and training uses only the terminal outcome. The result is a dense, outcome-anchored critic that outperforms PPO on math-reasoning RL without needing process-reward labels.

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [rlvr.md](./rlvr.md) · [reasoning/prm.md](./reasoning/prm.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md) · [_rewards.md](./_rewards.md)

---

## What it is

A drop-in replacement for PPO's value head on long-trajectory LLM RL. The critic head emits one scalar per token position `t`; these scalars **sum** into a sequence-level value logit `ℓ`, and the sigmoid `σ(ℓ)` is trained to match the terminal outcome `y ∈ {0, 1}`. The per-token scalars `v_t` are then interpretable as "how much did this token's content shift the log-odds of ending up right."

## How it works

**Bayesian decomposition in log-odds space.** Treat success as a binary event `y = 1`. By Bayes, after step `t`:

```
log P(y=1 | s_0..t) / P(y=0 | s_0..t) = log P(y=1) / P(y=0) + Σ_{k≤t} v_k
```

where `v_k = log [ p(s_k | y=1, s_<k) / p(s_k | y=0, s_<k) ]` is the step's log-likelihood ratio. The critic *parametrizes* each `v_k` with a per-token head on top of the hidden states.

**Training.** With rollouts of terminal outcome `y_i ∈ {0, 1}`, train the critic with binary cross-entropy on `σ(ℓ_i) = σ(Σ_t v_t^{(i)})`. No per-step labels needed.

**Advantage.** At inference time (RL step), compute `V(s_t) = σ(Σ_{k≤t} v_k) · R_max` and feed it as PPO's baseline. Alternatively, use the per-token `v_t` directly as a token-level advantage increment.

**RL objective.** Standard PPO clipped surrogate, plus a KL penalty to a reference model. Only the value head is reparametrized.

## Why it matters

- Fixes PPO's long-trajectory credit-assignment problem **without needing per-step process rewards**. The critic infers which tokens mattered from terminal supervision alone — a middle path between RLVR (outcome only) and PRM-style training (per-step labels).
- Reported on **Qwen3-4B-Base** and **Qwen3-8B-Base**: beats vanilla PPO on math reasoning benchmarks, with better out-of-distribution generalization and gains on non-math downstream tasks.
- Log-odds decomposition gives a **built-in interpretability affordance** — the per-token `v_t` are reward attributions you can inspect.

## Gotchas & tricks

- **Terminal-only supervision is a credit-assignment regularizer.** The critic can't put all the mass on one token; the decomposition forces it to spread credit across the trajectory. Good for stability, bad if a single token really is decisive.
- **Numerical range of `v_t`.** Unbounded `v_t` makes `ℓ` explode on long trajectories. Clip or tanh-squash the per-token head.
- **Compare to PRMs.** TPPO replaces the per-step label with per-step *inferred* contributions. If you *have* clean per-step labels, PRM-style training is likely still better.
- **Advantage computation.** Using `V(s_t)` as a baseline is the vanilla path. Using `v_t` as token-level advantages is more aggressive but can be unstable — the paper reports the former.
- **Reward signal.** TPPO is agnostic about the terminal reward: it works with RLVR, RM-based, or hybrid signals.

## Sources

- Paper: *TPPO: Tracking-Value PPO For Language Model Reasoning* — Li, Yan, Luo, Wang, Li, Zhan, Cheng, Lin — Zhejiang U. / Westlake / Shanghai AI Lab / HKUST / NJU / SJTU / NTU, 2026 — arXiv:2610.16794.
