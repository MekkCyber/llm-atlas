# NormGuard
*Depth — training-time hinge penalty that suppresses velocity-norm inflation under flow-matching RL.*

**TL;DR:** Across multiple RL post-training methods for flow-matching generators (NFT, AWM, DPO), the per-step velocity norm `‖v_θ‖` inflates by **5–15%** vs the reference. This drift is *co-adapted into the weights*, so inference-time renormalization (the standard fix in classifier-free guidance) fails. NormGuard is a **training-time hinge penalty** that activates only when `‖v_θ‖ > ‖v_ref‖`, composing additively with any velocity-local base loss.

**Prereqs:** [_rl](../post-training/_rl.md), [dpo](../post-training/dpo.md)
**Related:** [_rewards](../post-training/_rewards.md)

---

## What it is

A drop-in regularizer for RL post-training of flow-matching / rectified-flow generators. It addresses a specific failure mode where reward improves but perceptual quality (judged by an MLLM or by forensic-realism metrics) silently degrades, *because the reward proxy is blind to norm inflation*.

## How it works

**The diagnostic.** Across NFT, AWM, and DPO, `‖v_θ(x_t, t)‖` rises consistently relative to a frozen reference. An adjoint sensitivity analysis shows that velocity-norm magnitude carries **no coherent first-order reward signal at the batch level** — so suppressing it is unlikely to leave reward on the table.

**The penalty.** During training:

```
L_total = L_base + λ · max(0, ‖v_θ(x_t, t)‖ - ‖v_ref(x_t, t)‖)
```

The hinge means there is no gradient when `‖v_θ‖ ≤ ‖v_ref‖`; the penalty only fires on inflation. Additive composition means it stacks with any existing velocity-local objective (DPO, NFT, AWM, …).

**Why training-time, not inference-time.** Rescaling `v_θ` at inference (the CFG-style fix) **does not** recover quality once the inflation is co-adapted into the weights. The model has learned to *produce* the inflated velocity and to *depend on it* downstream.

## Why it matters

- **Quality-preserving RL for diffusion.** Across **2 bases × 3 methods × 2 reward proxies**, NormGuard improves MLLM-judged quality and forensic realism *without sacrificing reward*.
- **Effect amplifies at few-step inference**, where artifacts from inflated velocities compound.
- **Theoretically clean.** The adjoint-sensitivity argument explains why suppressing norm is *safe* (no reward signal to lose) rather than just empirically lucky.

## Gotchas & tricks

- Choose `λ` so the hinge bites in practice but doesn't dominate `L_base` — the paper's λ values are method-specific.
- The penalty needs the reference velocity `‖v_ref‖` per `(x_t, t)`; cache this for the SFT/reference checkpoint to avoid an extra forward at every step.
- Gains are **not** explained by early stopping; ablations rule this out.
- Useful complement to (not replacement for) reward-model improvements — addresses a different failure mode (norm drift) than reward hacking proper.

## Sources

- Paper: *NormGuard: Reward-Preserving Norm Constraints in Flow-Matching Reinforcement Learning* — Lianyu Pang et al., HKUST / Kuaishou — arXiv:2606.27771 — https://arxiv.org/abs/2606.27771
