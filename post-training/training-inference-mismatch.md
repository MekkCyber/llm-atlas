# Training–Inference Mismatch in RL

*Depth — when the rollout runs in a different numerical regime than the training step, the policy gradient the trainer sees doesn't match the policy that actually sampled the rollout.*

**TL;DR:** In LLM RL, rollouts and training steps often use different precisions (FP4/FP8 rollout vs BF16 train), different KV layouts, or different kernels. The result is that the sampling probability stored by the rollout differs from the probability the trainer recomputes — a *mismatch* that silently corrupts the policy-gradient estimate. The problem gets worse as the policy sharpens near convergence. Modern fixes decompose mismatch along the gradient direction and only correct the components that destabilize training.

**Prereqs:** [grpo.md](./grpo.md), [ppo.md](./ppo.md), [_rl.md](./_rl.md)
**Related:** [../quantization/fp4.md](../quantization/fp4.md), [../quantization/nvfp4.md](../quantization/nvfp4.md), [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

An LLM RL step is:

1. **Rollout:** sample `o ~ π_θ_old(· | q)` under *rollout* kernels/precision. Store sampled tokens and their *rollout* logprobs `log π_rollout(o_t)`.
2. **Train step:** recompute logprobs `log π_θ(o_t)` under *trainer* kernels/precision. Compute the importance ratio `r_t = exp(log π_θ − log π_rollout)` and apply PPO-clip.

If the two phases ran identical numerics, `log π_rollout = log π_θ_old` and the only quantity moving the ratio is the parameter update `θ_old → θ`. In reality:

- The rollout may run in FP4/FP8 for throughput; the trainer runs in BF16 or higher.
- Rollout batching, KV layout, and FlashAttention variants differ from training.
- Even at identical precision, kernel fusion / order of operations introduces floating-point differences.

So `log π_rollout(o_t) ≠ log π_θ_old(o_t)` even with `θ_old = θ`. The induced ratio drift is **not** policy-driven and should not count as a gradient signal — but PPO's clip does not know the difference.

## How it works

### Decomposing the mismatch

Let `Δ_t = log π_θ_old(o_t) − log π_rollout(o_t)` be the per-token mismatch. The PPO-ratio importance weight becomes:

```
r_t  =  exp(log π_θ(o_t) − log π_rollout(o_t))
     =  exp((log π_θ − log π_θ_old) + Δ_t)
     =  r_t^true  ·  exp(Δ_t)
```

So mismatch is a multiplicative perturbation to the true ratio. On a policy-gradient step, the direction of the resulting bias matters:

- When `Δ_t` is **co-aligned with the policy-gradient direction**, the update is *amplified* — the policy moves further than it should in exactly the direction of least evidence. This is the destabilizing component.
- When `Δ_t` is **anti-aligned**, the update is *contracted*. The step is smaller than intended but in the right direction. This is safe (just an effective learning-rate reduction on that token).

TRIAGE (2026) formalizes this decomposition and corrects only the amplifying component. Simpler mitigations (lower learning rate, importance-sample truncation) blunt both.

### Why it gets worse near convergence

Early in RL, policy entropy is high; typical `|log π_θ_old|` is modest and the mismatch is a small relative perturbation. Late in RL, the policy sharpens — some tokens have `log π_θ_old` approaching 0 and others approaching −∞. The same absolute mismatch `Δ_t ≈ 0.1` is tiny for a near-certain token and huge for a long-tail token. Mismatch "bites" exactly where the policy is already brittle.

### Where it comes from in practice

| Source | Typical magnitude | Mitigation |
|---|---|---|
| FP4 rollout, BF16 train | up to several logits/token on sharp tokens | TRACE (RGQAT), TRIAGE (direction-aware), or match rollout and train precision |
| FP8 rollout, BF16 train | ~0.01–0.1 logits/token | often ignorable; LR scheduling handles it |
| Different FlashAttention variants | ~0.001–0.01 logits/token | fuse rollout and train kernels |
| Different kernels (vLLM vs Megatron) | engineer-dependent | log and verify ratio drift at init |

## Why it matters

- **Enables aggressive rollout acceleration.** FP4 rollouts are 2–5× faster than BF16 and halve the dominant cost in LLM RL — but they inflate mismatch. A principled fix is what makes FP4 RL safe end-to-end.
- **Shows up as "policy collapse near convergence"** in practice. Teams who report RL training that works until step N and then flatlines or diverges often have an undiagnosed training–inference mismatch problem.
- **A reusable diagnostic.** Logging `max_t |Δ_t|` and the per-batch mean of `exp(Δ_t) − 1` is cheap and catches most problems before they bite.

## Gotchas & tricks

- **Measure before mitigating.** Record `Δ_t` distributions across a few batches. If they're centered on zero with small spread, you don't have a mismatch problem; if they drift or widen with training, you do.
- **Importance clipping truncates both the signal and the mismatch.** It's a blunt but universally effective fix; TRIAGE / direction-aware methods are strictly better when you can afford them.
- **Prefer matching precision over correcting bias.** If possible, run rollout and training at the same precision. All mitigations degrade gracefully if you can't.
- **Mismatch is not just logit drift.** Different top-k truncation, different sampling temperature cast order, different RNG state can all contribute; all should be audited together.

## Sources

- Paper: *TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning* — Zhang et al., 2026 — [arXiv:2610.07043](https://arxiv.org/abs/2610.07043).
- Paper: *TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models* — Wang et al., 2026 — [arXiv:2610.07767](https://arxiv.org/abs/2610.07767).
- Foundational: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017 — PPO's importance-ratio surgery that this problem lives inside.
