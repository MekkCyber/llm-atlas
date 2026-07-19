# Diffusion RL
*Depth — reward-driven fine-tuning of diffusion and flow-matching generators.*

**TL;DR:** Apply policy-gradient-style RL to diffusion or flow-matching generators to fine-tune them against a reward signal (aesthetic score, human preference, task success). Two flavors dominate: *reverse-process RL* (treat the sampling chain as an MDP and update along it — DDPO, DPOK) and *forward-process RL* (compute the update directly on the score/velocity network — DiffusionNFT and successors). Recent work (MeanFlowNFT) extends forward-process RL to few-step *average-velocity* generators via an induced predictor.

**Prereqs:** [../post-training/_rl](../post-training/_rl.md), [../post-training/ppo](../post-training/ppo.md)
**Related:** [README](README.md), [../post-training/dpo](../post-training/dpo.md)

---

## What it is

Diffusion generators produce samples by iteratively denoising from Gaussian noise. Standard training uses a denoising-loss objective. Diffusion RL replaces (or complements) that with a reward-driven objective, so the model is fine-tuned to maximize a scalar reward `r(x)` over its generated samples — analogous to RLHF over LLMs, but for generative-vision models.

## How it works

### Reverse-process RL

Treat the sequence of denoising steps as an MDP: the state is the intermediate noisy sample, the action is the sampled next state, the reward is `r(x)` at the final sample (backed up along the chain). Apply PPO or REINFORCE with a variance-reduction baseline. DDPO and DPOK are the canonical instantiations.

Pros: clean policy-gradient interpretation, works with any pretrained diffusion. Cons: high variance from the long sampling chain, and the update pattern is coupled to the sampler.

### Forward-process RL

Compute the RL update directly on the score / velocity network at the training-time noise level, avoiding the long sampling chain. DiffusionNFT (Diffusion Noise-conditioned Fine-Tuning) is the anchor: reward-weight the standard denoising loss at each noise level so higher-reward samples steer the score network more.

Pros: much lower variance, updates fit cleanly into the standard training loop. Cons: needs the reward to be evaluable on partially-denoised (or generated) samples in a way that back-propagates cleanly.

### Bridging to few-step generators

MeanFlow parameterizes an **average-velocity** network `v̄(x, t₀, t₁)` for fast few-step sampling — it directly predicts the mean flow over a whole interval, not the instantaneous velocity. DiffusionNFT is defined on instantaneous velocity, so it does not apply directly.

MeanFlowNFT (2026) resolves this with an **induced predictor**: derive an instantaneous-velocity view of `v̄` implicitly, apply the DiffusionNFT update to that view, and prove the policy-improvement guarantee carries back to the deployed average-velocity network. The result is few-step (4-step) generators that can be reward-optimized without inflating the sampling budget.

## Why it matters

- **Aligns generative vision with generative language.** RLHF-style fine-tuning became standard for LLMs years before diffusion — diffusion RL closes the gap for image and video generation.
- **Unlocks reward-optimized fast samplers.** Combining few-step generators with RL fine-tuning was the missing piece for cost-conscious deployment of aesthetic- or preference-tuned image/video models.
- **Reduces reliance on labeled data.** A trained reward model or automatic scoring function can substitute for large pairwise-preference datasets.

## Gotchas & tricks

- **Reward hacking is severe.** Aesthetic scores in particular have obvious exploits (over-saturated color, high-frequency noise). Clip reward, ensemble reward models, or add a KL-to-base regularizer.
- **Base-model drift.** Long RL runs push the generator off the pretrained manifold, degrading diversity. Early stopping and KL regularization to the base help.
- **Reward evaluation cost.** Reward evaluated on the final sample is what you want, but for reverse-process RL you often need per-step estimates — either learn a per-step reward or accept the terminal-only sparse signal.
- **Sampler coupling for reverse-process RL.** The update is tied to the specific sampler used at training; deploying with a different sampler at inference can undo gains.

## Sources

- Paper: *DDPO: Training Diffusion Models with Reinforcement Learning* — Black et al., 2023 — reverse-process RL for diffusion.
- Paper: *DPOK* — Fan et al., 2023 — companion policy-gradient formulation.
- Paper: *DiffusionNFT: Noise-conditioned Fine-Tuning* — 2025 — forward-process RL for diffusion.
- Paper: *MeanFlowNFT: Bringing Forward-Process RL to Average-Velocity Generators* — Huang et al., 2026 (Tencent Hunyuan / HKUST) — induced-predictor extension to few-step MeanFlow.
