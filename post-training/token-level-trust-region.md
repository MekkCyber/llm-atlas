# Token-level trust region (CPPO)

*Depth — making PPO's clip bound per-token instead of uniform.*

**TL;DR:** PPO and GRPO use the same clip $\epsilon$ at every token in a trajectory. But long-CoT trajectories have *wildly heterogeneous* tokens — high-entropy reasoning forks where the policy needs to move, and low-entropy boilerplate where it shouldn't. CPPO (Contextual PPO) makes the clip bound a function of the token's context — wider at high-value tokens, tighter at routine tokens — and reports more stable RLVR training across model scales.

**Prereqs:** [ppo](ppo.md), [grpo](grpo.md), [rlvr](rlvr.md)
**Related:** [_rl-trust-regions](_rl-trust-regions.md) · [long-cot-rl](reasoning/long-cot-rl.md) · [token-credit-assignment](reasoning/token-credit-assignment.md)

---

## What it is

The standard PPO clip $\mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)$ uses the same $\epsilon$ at every token position. CPPO replaces the constant with a *contextual* function $\epsilon_t = f(\text{position}, \text{entropy}, \text{advantage}, \ldots)$, so the trust region widens where the policy genuinely needs to update and tightens where it doesn't.

The pathology this fixes: in long-CoT RL, most tokens have very low gradient signal (deterministic continuations like "Let me think step by step"). A few tokens are *reasoning forks* — branch points where the policy actually changes the trajectory. A uniform $\epsilon = 0.2$ is too tight for the forks (clipping kills useful exploration) and too loose for the boilerplate (allowing irrelevant drift). CPPO's per-token width fixes both.

## How it works

For each token $t$ in response $o_i$:

1. Compute a *per-token signal* — the paper draws this from token entropy, advantage magnitude, and position within the CoT. (Implementation details: high entropy = informative reasoning fork → wider $\epsilon$; high $|A|$ = strong gradient incentive → wider $\epsilon$.)
2. Map the signal to a clip width via a monotonic schedule: $\epsilon_t = \epsilon_\text{base} \cdot g(\text{signal}_t)$, where $g$ widens for high-entropy / high-advantage tokens.
3. Apply the PPO objective with $\epsilon_t$ in place of the constant:
   $$ L^\text{CPPO}_t = \min\left(r_t A_t,\; \mathrm{clip}(r_t, 1-\epsilon_t, 1+\epsilon_t) A_t\right) $$

The downstream pipeline (group-mean baseline for GRPO, KL-to-ref penalty, etc.) is unchanged — CPPO is a drop-in replacement for the clip term only.

## Why it matters

- **Long-CoT RL stability.** Uniform clipping is the most-tuned hyperparameter in RLVR. Making it context-aware removes one of the brittle knobs.
- **Composes with GRPO.** No new components; per-token $\epsilon_t$ replaces the constant in the existing loss.
- **Same direction as DAPO's asymmetric clip.** DAPO opens the *upper* clip bound to encourage positive-advantage exploration. CPPO opens it per-token based on token role. The two are compatible.

## Gotchas & tricks

- **Signal definition matters.** Entropy + advantage are the obvious choices; the paper picks a specific functional form and ablates a few variants. Reproductions should not assume any one signal — pick what your trajectories actually need.
- **Calibrate against a tuned GRPO baseline.** A poorly-tuned GRPO with uniform $\epsilon = 0.2$ is easy to beat. The CPPO gains should be measured against a careful baseline sweep over $\epsilon$.
- **Don't widen $\epsilon$ too far.** Wide $\epsilon$ on a noisy token is worse than narrow $\epsilon$ on the same token — once you're outside the trust region, you're optimizing a bad surrogate.
- **Per-token width interacts with KL-to-ref.** The KL term is unchanged but its effective contribution shifts when many tokens get wider $\epsilon$. Re-tune $\beta_\text{KL}$ when switching.

## Sources

- Paper: *Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning* — Mao et al., Tencent Hunyuan, 2026 — [arXiv 2606.10968](https://arxiv.org/abs/2606.10968).
- Background: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017.
- Related: *DAPO* — Yu et al., 2025 — asymmetric clip motivated by the same problem.
