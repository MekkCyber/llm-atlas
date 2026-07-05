# Distribution-wise Rewards

*Depth — a reward class for visual-generative RL that scores the generated *set* against the real distribution, not each sample independently.*

**TL;DR:** Standard RL fine-tuning of visual generators uses **sample-wise** rewards: score each generated sample independently, sum the gradients. This lets the policy hack a single high-reward mode and collapse diversity. **Distribution-wise rewards** score the *contribution of the new sample to the divergence between the generated distribution and the reference*, penalizing samples that duplicate what the policy already produces. A **subset-replace** trick avoids re-estimating the divergence over a huge reference each step. FID-50K improves 8.30 → 5.77 on SiT and 3.74 → 3.52 on EDM2 while preserving diversity.

**Prereqs:** [_rewards.md](_rewards.md), [_rl.md](_rl.md)
**Related:** [rlvr.md](rlvr.md), [../pre-training/model-souping.md](../pre-training/model-souping.md)

---

## What it is

A drop-in replacement for the sample-wise reward function used in RL post-training of visual generators. Instead of $R(x_i)$, the reward is $R(x_i;\, \mathcal{G})$ — the marginal effect of adding $x_i$ to the current generated set $\mathcal{G}$ on some divergence to the real-data distribution. Samples that push the generated distribution *closer* to real are rewarded; samples that duplicate existing modes are not.

## How it works

### The distribution-level reward

Let $p_{\text{real}}$ be the target distribution and $\mathcal{G}$ the current generated reference set. For a new candidate $x$, the reward is roughly

$$
R(x;\, \mathcal{G}) = D(\,p_{\text{real}} \;\|\; \mathcal{G}\,) - D(\,p_{\text{real}} \;\|\; \mathcal{G} \cup \{x\}\,)
$$

for some divergence $D$ (an FID-style or feature-distance metric). If $x$ moves the generated distribution toward real, $R(x;\mathcal{G}) > 0$; if $x$ is a duplicate of an over-represented mode, $R(x;\mathcal{G}) \approx 0$ or negative.

### Subset-replace to make it tractable

Recomputing the divergence over a large reference set each step is prohibitive. The **subset-replace** trick maintains a small representative subset of $\mathcal{G}$ and only updates a fraction of it per RL step — the marginal contribution of $x$ is estimated against this rolling subset, drastically reducing the per-sample cost while keeping the reward faithful.

### Post-hoc merging coefficients as an RL variable

Regular RL with SDE samplers has a train-inference mismatch (training uses stochastic ODE with different noise than inference). The paper additionally uses RL to optimize **post-hoc model-merging coefficients** between checkpoints, letting the objective smooth the mismatch. See [../pre-training/model-souping.md](../pre-training/model-souping.md) for the merging background.

## Why it matters

- **Direct mechanical fix for mode collapse.** Sample-wise reward + RL on diffusion has been a mode-collapse machine. Distribution-wise reward makes the pathology impossible by construction — a mode-hit gives no reward once the mode is saturated.
- **Cheap enough to deploy.** The subset-replace estimator makes distribution-level rewards fit inside a normal RL step budget. Prior work on distribution-matching had this idea in principle but no tractable implementation.
- **Applies beyond images.** Nothing in the reward class is image-specific. Text and video generators show the same mode-collapse pathologies under sample-wise RL; distribution-wise rewards plausibly transfer.

## Gotchas & tricks

- **Subset size is the knob.** Too small and the estimate is noisy (reward is essentially random); too big and you lose the compute savings. Start at ~1000 and adjust for reward stability.
- **Divergence choice matters.** FID-style feature-space divergences track perceptual quality; MMD in a raw pixel space doesn't. Pick a divergence that correlates with the property you actually care about.
- **Warm-start the reference $\mathcal{G}$.** An empty $\mathcal{G}$ makes every early sample "novel," which mimics sample-wise reward at start of training. Pre-populate from a small held-out reference sample.
- **KL to a reference model is still needed.** The distribution-level reward doesn't prevent the policy from drifting off-distribution in ways the divergence can't see. Keep the KL term.

## Sources

- Paper: *Optimizing Visual Generative Models via Distribution-wise Rewards* — 2026 — [arXiv:2607.02291](https://arxiv.org/abs/2607.02291). Reports SiT FID 8.30 → 5.77 and EDM2 FID 3.74 → 3.52.
