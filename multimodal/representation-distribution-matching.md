# Representation Distribution Matching (RDM) for One-Step Generation
*Depth — training a one-step image generator by matching generated- and reference-feature distributions under frozen encoders.*

**TL;DR:** Multi-step diffusion generators can be distilled into **one-step** samplers in many ways: score distillation, consistency models, adversarial matching. **RDM** is a fourth: feed a batch of generated samples and a batch of real samples through a frozen pretrained encoder, and train the generator to minimize a distributional distance between their feature distributions. The **iRDM** design-space study finds three keys: (1) *big generated batches* (>2048) make MMD strong again; (2) match against a *battery* of encoders, not one; (3) evaluate with SW_r^14 — a Sliced-Wasserstein over 14 disjoint encoders — to avoid gaming. iRDM sets one-step SOTA on ImageNet and post-trains FLUX.2 [klein] to a one-step generator in 90 H200 GPU-hours.

**Prereqs:** [../evaluation/README.md](../evaluation/README.md)
**Related:** [../post-training/distribution-wise-rewards.md](../post-training/distribution-wise-rewards.md)

---

## What it is

Score distillation and consistency models distill a multi-step teacher into a one-step student by matching *per-sample* signals (scores, or trajectory endpoints). RDM is a *distributional* alternative: don't match individual samples, match the *distribution* of features that generated batches induce against the distribution real batches induce, under frozen pretrained encoders.

The classical instantiation — matching MMD (Maximum Mean Discrepancy) between generated and real batches under a single encoder — was tried a decade ago and abandoned as insufficient for high-quality generation. iRDM revisits this and shows the technique was under-scaled and under-diversified, not fundamentally broken.

## How it works

**Setup.** A generator $G_\theta$ produces a one-step image from noise. A battery of frozen pretrained encoders $\{E_k\}$ maps images to features.

**Loss.** For a batch of generations $\{x_i\}$ and a batch of reals $\{y_j\}$:

$$
\mathcal{L}_\text{RDM} = \sum_{k} D_k\big(\{E_k(x_i)\}, \{E_k(y_j)\}\big)
$$

Where $D_k$ is a distributional metric — MMD with a well-tuned kernel is the paper's finding. Backprop through $E_k$ to $G_\theta$.

**Three design axes established:**
1. **How you compare distributions.** MMD wins if the *generated batch* is large enough — optimum above 2048 samples per step, far beyond conventional batch sizes.
2. **What you compare in.** A single encoder gets gamed — the generator learns to push $E_k$'s feature statistics into a match while producing visibly fake images. Matching against a *balanced battery* of encoders (different architectures, different pretraining) prevents this.
3. **How you evaluate.** SW_r^14 — a Sliced-Wasserstein distance over 14 encoders **disjoint from training** — is used as an evaluation-loss decoupling: the generator can't overfit the objective by construction.

## Why it matters

- **One-step generation, cheaply.** iRDM sets a new one-step SOTA on ImageNet at SW_r^14 = 1.30. Post-training FLUX.2 [klein] from four steps to one step surpasses the four-step version on GenEval (0.826 vs 0.794) in 90 H200 GPU-hours.
- **Human-preference alignment as a side effect.** PickScore — a human-preference proxy not in the training loss — prefers iRDM over the prior best one-step generator on **71.2 %** of matched samples.
- **A general "train-vs-eval decoupling" recipe.** The battery-of-encoders + Sliced-Wasserstein-eval pattern applies beyond one-step generation to any distributional-loss training where reward hacking is a concern.

## Gotchas & tricks

- **Batch size dominates.** Under 2048 generated samples per step, MMD is noisy and iRDM underperforms. Above 2048, gains scale slowly. Budget accordingly.
- **Encoder diversity matters more than encoder count.** Two very different encoders beat five similar ones. Mix conv nets, vision transformers, DINO-style self-supervised nets.
- **Evaluation must use disjoint encoders.** Using training encoders to evaluate re-opens gaming. SW_r^14 with unrelated encoders is what makes the metric honest.
- **The generator sees no per-sample supervision.** Everything is at the batch/distribution level. Failure modes look like batch-averaged artifacts, not per-image errors.

## Sources

- Paper: *Representation Distribution Matching for One-Step Visual Generation* — Feng et al., 2026 — [arXiv:2607.02375](https://arxiv.org/abs/2607.02375).
- Related: *Adversarial Score Distillation / Score-based generative models* — the per-sample distillation baselines.
- Historical: *Generative Moment Matching Networks* — Li et al., 2015 — the original MMD-generator idea RDM revisits at scale.
