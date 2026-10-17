# Routing Policy Optimization (RPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RL for diffusion generators usually injects noise into the reverse denoising trajectory (Flow-GRPO et al.) for exploration, which forces a stochastic SDE sampler during training and pays rollout cost on every noisy trajectory. **RPO** replaces that with **sparse attention routing as the stochastic action**: at a few selected denoising levels, each query block samples a subset of key blocks proportional to the generator's own attention mass. At inference, routing is removed and the original deterministic sampler is kept.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [rlvr.md](./rlvr.md)

---

## What it is

Diffusion RL maximizes a reward (CLIP score, aesthetic model, task-specific signal) under a KL to the pretrained generator. The score-function estimator needs an action distribution to differentiate. Prior methods treat the per-step velocity noise as the action, which turns the deterministic ODE sampler into an SDE for training.

RPO's observation: the deterministic sampler already *has* a natural stochastic action inside it — **which key tokens each query attends to**. Sparsify attention stochastically at a few timesteps, score the perturbation, and learn.

## How it works

### Attention routing as the action

At chosen timesteps $t \in \mathcal{T}_{\text{route}}$, for each query block $q$:

- Keep $q$'s own block (self-anchor).
- Sample a subset of key blocks with probabilities proportional to the generator's **native attention mass** (teacher-forcing-free: use the actual dense softmax values the model would produce).
- Alternatively "keep dense" with some fallback probability.

Un-routed timesteps keep dense attention. **Dense anchor layers** (early-stack blocks that must see everything) are excluded from routing.

### Velocity perturbation and Gaussian likelihood

The sampled route changes the velocity $v_\theta$ at that timestep by some $\delta v$. Model the perturbation as $\delta v \sim \mathcal{N}(0, \Sigma)$ around the dense velocity. The log-likelihood of the sampled route under this Gaussian is a differentiable function of the generator's dense-attention outputs — so updates flow back through $v_\theta$ even after routing is removed.

### Two-rollout advantage estimator

For each prompt, run **two rollouts** that share:
- the same prompt,
- the same initial latent,
- the same routed timesteps,
- the same dense anchor layers,

and differ only in their **sampled routes**. Reward difference between the two rollouts gives an unbiased advantage for the routed transitions, controlling for everything else. This is the GRPO group-baseline trick specialized to pairs.

### Inference is unchanged

At test time, routing is dropped, the original deterministic ODE sampler runs. Because the Gaussian likelihood is defined around the dense velocity, the learned updates target the dense generator; removing routing does not throw them away.

## Why it matters

- **~4× training efficiency over Flow-GRPO-style.** On Stable Diffusion 3.5 Medium and Wan2.2, RPO reaches comparable or better quality at a quarter of the training cost.
- **Keeps the deterministic sampler.** No test-time SDE, no sampler change — the model can slot into existing pipelines unmodified.
- **Reuses the model's own prior.** Sampling routes proportional to native attention mass means the exploration distribution is already peaked where the generator thinks the signal is; much lower variance than isotropic noise.

## Gotchas & tricks

- **Routed timesteps are a hyperparameter.** The paper uses a small sparse set; routing at every step inflates the stochasticity budget and hurts quality.
- **Anchor layers matter.** Routing a layer that controls global coherence (early DiT blocks) degrades image structure; the paper keeps those dense.
- **Two-rollout, not many-rollout.** The pair design is cheap but gives one advantage per prompt — if your reward is noisy you may want more pairs rather than more routes per pair.
- **Reward model still the bottleneck.** RPO cuts rollout cost; the reward call cost is unchanged. For expensive judges (large VLM scorers) RPO's win is proportional to the generator-vs-judge ratio.
- **Text-to-video evaluated too.** Not just T2I — Wan2.2 numbers show the trick carries to the longer temporal rollouts where SDE-based diffusion RL is especially expensive.

## Sources

- Paper: *Routing Policy Optimization for Online Diffusion Reinforcement Learning* — Shen, Yang, Zhang, Huang, Lin, Liu, Kang, Yang, Wang, Ma, Wang (NUS + ByteDance), 2026 — [arXiv:2610.22282](https://arxiv.org/abs/2610.22282).
- Project page: https://velox-world.com/rpo
- Baseline family: Flow-GRPO and DDPO (score-function diffusion RL via trajectory noise).
