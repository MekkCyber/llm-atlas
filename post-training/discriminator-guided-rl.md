# Discriminator-Guided RL (DRL)

*Depth — use a pretrained-space discriminator's logit as the reward in KL-regularized RL fine-tuning of flow / score-matching models.*

**TL;DR:** Score- and flow-matching losses regress on velocity fields under training-time marginals — a proxy that's poorly aligned with the visual / semantic properties that actually determine sample quality at inference. Discriminator-Guided RL closes the gap: train a discriminator to separate **real data** from **base-model samples** in a *pretrained representation space* (e.g. DINOv3), and use its **logit as the reward** in KL-regularized RL fine-tuning. The pretrained space restricts the discriminator to perceptually meaningful directions; the logit estimates the log-likelihood ratio between data and model — provably the optimal reward for matching the data distribution.

**Prereqs:** [_rl](_rl.md), [_rewards](_rewards.md)
**Related:** [grpo](grpo.md), [ppo](ppo.md)

---

## What it is

A clean replacement for preference-based RL in generative-model fine-tuning. Instead of asking humans whether sample A or B is better (expensive, conflated with annotator taste, conflicts with realism), DRL gets a reward directly from a discriminator that learned to tell data from samples.

## How it works

Given a pretrained generative model $p_\theta$ (SiT, JiT, REPA, RAE — diffusion / flow-matching), and a real-data dataset $\mathcal{D}$:

1. **Pick a representation space.** Use a pretrained vision encoder $\phi$ (e.g. DINOv3) to map images to features.
2. **Train discriminator** $D_\psi$: a small MLP on top of $\phi$, trained with standard binary cross-entropy to separate $\phi(x), x \sim \mathcal{D}$ (real) from $\phi(\tilde{x}), \tilde{x} \sim p_\theta$ (base-model samples).
3. **Reward** for a generated $\tilde{x}$:
   $$r(\tilde{x}) = \text{logit}(D_\psi(\phi(\tilde{x}))) = \log \frac{D_\psi}{1 - D_\psi}$$
   This logit, by Bayes' rule, estimates $\log \frac{p_{\text{data}}(\tilde{x})}{p_\theta(\tilde{x})}$ — exactly the reward needed to push $p_\theta$ toward $p_{\text{data}}$ under KL-regularized RL.
4. **KL-regularized RL update** on $\theta$:
   $$\nabla_\theta \mathbb{E}_{p_\theta}[r(\tilde{x})] - \beta \nabla_\theta D_\text{KL}(p_\theta \| p_{\theta_0})$$
   where $p_{\theta_0}$ is the base model — standard RLHF objective, just with the discriminator-derived reward.

The KL prevents the policy from collapsing to discriminator-favorite modes; the pretrained space restricts the reward landscape to directions that matter perceptually.

## Why it matters

- **No human preferences.** Avoids the expense and annotator-taste confounds of preference data while still getting an RL signal that correlates with human judgment.
- **Big concrete gains.** On SiT: guidance-free FID **9.38 → 2.62**, semantic-space FD on DINOv3 **88.2 → 19.3**. Comparable gains on JiT, REPA, RAE. Human-preference rewards improve too, *without* training on them.
- **Composes with preference RL.** DRL → preference RL gives a better Pareto frontier than either alone — DRL fixes the data-distribution match, preference RL fixes the alignment dimension.
- **Principled fix.** Score / flow matching's "loss ≠ quality" mismatch has been papered over with classifier-free guidance for years. DRL is the first reward-based fix grounded in the data distribution itself.

## Gotchas & tricks

- **Discriminator overfitting.** A discriminator that perfectly separates data from samples gives a useless reward (saturated logits). Standard GAN regularization tricks (R1, spectral norm) apply.
- **Pretrained-space choice matters.** DINOv3 captures semantic + low-level realism; CLIP captures semantic + text-alignment. Pick the space for the gap you're targeting.
- **Reduces oversaturation / excessive brightness** as a side effect — the discriminator notices these are off-data even when CFG doesn't.
- **Doesn't help with subjective preferences** (style, taste). That's still preference-RL territory; DRL handles realism + structural correctness.

## Sources

- Paper: *The Reward Was in Your Data All Along: Correcting Flow Matching with Discriminator-Guided RL* — Beltran-Velez, Friedrich, Xiaofeng, Askari-Hemmat, Han, Romero-Soriano, Drozdzal et al., Meta AI, 2026 — [arXiv:2606.19162](https://arxiv.org/abs/2606.19162).
