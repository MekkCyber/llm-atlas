# Predictive Divergence Masks (PDM)

*Depth — a sample-free surrogate for the importance ratio in off-policy LLM RL.*

**TL;DR:** Practical LLM RL (PPO, GRPO) is always slightly off-policy because of training/inference mismatch and policy staleness. The standard fix is a **sampled importance ratio** $\pi_\theta / \pi_{\theta_\text{old}}$, clipped to a window. Predictive Divergence Masks replace that noisy ratio with a **predictive check**: compute the *directional derivative* of the divergence between current and behaviour policies at each token, mask out the update wherever taking the gradient step would increase divergence. Sample-free, no clip-range hyperparameter.

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [reasoning/online-policy-mirror-descent.md](./reasoning/online-policy-mirror-descent.md) · [rlvr.md](./rlvr.md)

---

## What it is

PPO's clipped ratio $\text{clip}(\pi_\theta/\pi_{\theta_\text{old}}, 1-\epsilon, 1+\epsilon)$ is a *reactive* fix: sample a token, compute the ratio, discard the update if the ratio wanders too far. It works but is high-variance (single-sample ratio) and hyperparameter-sensitive (clip range $\epsilon$).

PDM asks the *predictive* question instead: **would applying this gradient step increase $D_\text{KL}(\pi_\theta \| \pi_{\theta_\text{old}})$?** If yes, mask the update at that token. The answer comes from a closed-form directional derivative — no ratio sampling, no clip window.

## How it works

At each token position $t$ with proposed gradient $\nabla_\theta \mathcal L_t$, PDM computes the sign of

$$
\frac{d}{d\eta} \, D_\text{KL}\!\left(\pi_{\theta - \eta \nabla_\theta \mathcal L_t} \,\Big\|\, \pi_{\theta_\text{old}}\right)\Bigg|_{\eta = 0}
$$

which reduces analytically to an inner product between the loss gradient and the divergence gradient at token $t$. Where that sign is positive (the step would drive $\pi_\theta$ away from $\pi_{\theta_\text{old}}$), the mask is zero; otherwise the mask is one. The masked gradient is then applied.

Because the derivative is evaluated *analytically*, there is no sampling variance and no clip window to tune — the mask is a deterministic, per-token binary.

## Why it matters

- **Kills the clip-range hyperparameter.** $\epsilon$ tuning is one of the more thankless steps in scaling GRPO/PPO runs; PDM removes it.
- **Lower gradient variance than sampled importance ratios.** The directional derivative is exact given the current parameters; the importance ratio is a single-sample estimate.
- **Composable with existing pipelines.** It's a per-token mask on the surrogate loss, i.e. a one-line change on top of GRPO/PPO.
- **Nice diagnostic.** The fraction of masked tokens is a cheap monitor for how off-policy the current batch has drifted — a signal today's stacks compute only indirectly via mean ratio.

## Gotchas & tricks

- **Directional, not magnitude.** PDM only masks in/out; it does not down-weight. Papers combining PDM with a shrunk step size may need to add magnitude control separately.
- **The mask is per-token but the reward is per-response.** Under GRPO's response-level advantage, PDM's mask determines *which tokens' gradients survive* for a given advantage $A_i$; a heavily-masked response contributes proportionally less gradient.
- **Behaviour policy identity matters.** For pipelines with rolling reference models, the choice of $\pi_{\theta_\text{old}}$ (last step vs last epoch vs SFT ref) changes what the mask conserves. Match it to what the rest of the loss (KL penalty, clip target) uses.
- **Numerical stability.** The inner product involves log-prob gradients that can be sharp near saturated softmaxes; standard log-softmax stabilisation is prerequisite.

## Sources

- Paper: *Predictive Divergence Masks for LLM RL* — Zhou, Yao, Qi, Ping, Tang, Wang, Pang — Tencent Hunyuan / UIUC / NUS, 2026 — introduces the analytic mask and its GRPO drop-in.
- Companion primitives: PPO's clipped ratio ([ppo.md](./ppo.md)); GRPO's response-level advantages ([grpo.md](./grpo.md)); KL-regularised mirror-descent framing ([reasoning/online-policy-mirror-descent.md](./reasoning/online-policy-mirror-descent.md)).
