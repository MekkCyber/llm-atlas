# Contrastive Policy Optimization (CPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RLVR pipelines routinely use **entropy** as an advantage-shaping signal (bonus for uncertain tokens, penalty for confident ones). Entropy conflates *useful* uncertainty (exploration) with *detrimental* confusion (broken reasoning). **CPO** replaces entropy with a token-level **contrastive disagreement** between reference-guided and vanilla generation distributions — a correctness-aware signal that distinguishes "uncertain but on-track" from "confused".

**Prereqs:** [rlvr.md](rlvr.md), [grpo.md](grpo.md), [_rewards.md](_rewards.md)
**Related:** [dpo.md](dpo.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

Advantage shaping in RLVR is the practice of adding per-token bonuses/penalties to the group-relative advantage before the PPO-clipped update:

$$A_t^{\text{shaped}} = A_i + \lambda \cdot s_t$$

where $A_i$ is the response-level advantage (GRPO-style) and $s_t$ is a per-token shaping signal. Entropy is the usual choice: $s_t = -H(\pi_\theta(\,\cdot\, \mid q, o_{<t}))$, encouraging exploration on uncertain tokens.

The trouble: entropy is *symmetric* over "I'm exploring" and "I'm broken." A token where the model is uncertain because it's actually reasoning ambiguously benefits from exploration; a token where it's uncertain because the CoT went off the rails should be penalized, not encouraged.

CPO's shaping signal is different: it's the disagreement between two conditional distributions on the same token position — one produced with a reference guide, one without. If they agree, the token is "on-track"; if they diverge, the token is "off-track" and needs correction.

## How it works

For each token position $(q, o_{<t})$:

1. **Vanilla distribution.** $\pi_\theta(\,\cdot\, \mid q, o_{<t})$ — the current policy's next-token distribution.
2. **Reference-guided distribution.** $\pi_\theta^{\text{ref}}(\,\cdot\, \mid q, o_{<t})$ — the same policy but conditioned on additional reference material (a canonical reasoning trace, a hint, a verifier's partial signal). Implementation-specific; the paper describes the reference signal used.
3. **Contrastive disagreement.**
   $$s_t = D\big(\pi_\theta(\,\cdot\, \mid q, o_{<t}) \,\|\, \pi_\theta^{\text{ref}}(\,\cdot\, \mid q, o_{<t})\big)$$
   for some divergence $D$ (KL or JS). Large disagreement ⇒ the vanilla generation is diverging from what the guided version would do — probably off-track.
4. **Shape the advantage.**
   $$A_t^{\text{CPO}} = A_i - \lambda \cdot s_t$$
   Positive advantage tokens with low disagreement (on-track) get amplified; positive-advantage tokens with high disagreement get muted. Negative-advantage tokens likewise.
5. **Standard PPO-clipped update.** Everything else is GRPO / RLVR as usual.

## Why it matters

- **Drop-in for any RLVR/GRPO pipeline.** Same loss shape, different shaping signal.
- **Corrects a widespread signal-mixup.** Entropy has been the default for a year of RLVR work; if a correctness-aware signal is cheap, it's a broadly applicable upgrade.
- **Reference-guided disagreement is a process-reward proxy.** Aligns with the direction of process-reward-model (PRM) research but without training a separate PRM — the reference-guided rollout is the ground truth.
- **Compositional with other shaping.** Doesn't preclude combining with entropy bonuses, length penalties, or format rewards.

## Gotchas & tricks

- **Reference-guided cost.** Each token needs an extra forward pass through the reference-guided setup. Cheap if the reference is a short prefix; expensive if it requires a full reranking.
- **Choice of $D$ matters.** KL is asymmetric; JS is symmetric but bounded. Try both; the paper's default is spelled out there.
- **$\lambda$ needs tuning.** Too small: shaping is noise. Too large: the shaping term dominates and the policy over-corrects to look like the reference.
- **Requires a meaningful reference signal.** If the "reference-guided" version is barely different from vanilla (weak reference), the signal is uninformative.
- **Not a silver bullet for reward hacking.** CPO makes advantage shaping more informative but doesn't fix rewards that were miscalibrated in the first place.

## Sources

- Paper: *Beyond Entropy: Correctness-Aware Advantage Shaping via Contrastive Policy Optimization* — Xu, Liu, Chan, Li, Cai, Chen, Zhang — CUHK / SCUT / NTU, 2026 — [arXiv:2607.14614](https://arxiv.org/abs/2607.14614).
- Foundational: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017 — the PPO baseline CPO reshapes.
- Related: process-reward models — [reasoning/prm.md](reasoning/prm.md) — same goal, different mechanism.
