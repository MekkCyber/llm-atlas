# Posterior Matching Distillation (PMD)
*Depth — few-step diffusion distillation that matches an exponentially-reweighted average of student samples to the teacher's posterior mean.*

**TL;DR:** A new objective for distilling diffusion models into 4-step (or fewer) generators. Instead of matching the student to the teacher pointwise along the trajectory, PMD matches an **exponentially-weighted running average of student samples to the teacher's posterior mean**. Under Gaussian corruption, each training target reduces to **one frozen-teacher evaluation at a shifted input** — cheap per step. Best FID among 4-step methods on SDXL and FLUX.2-Klein-4B with improved intra-prompt diversity.

**Prereqs:** [README](README.md)
**Related:** [../post-training/_post-training](../post-training/_post-training.md)

---

## What it is

Few-step diffusion distillation turns an $N$-step (say 50-step) teacher into a student that produces comparable images in $K \ll N$ steps (typically 1–4). Prior methods:

| Family | Target |
| --- | --- |
| Consistency distillation | Match student output to teacher output along the reverse trajectory |
| Score distillation (SDS, Diff-Instruct) | Match score / gradient fields |
| Progressive distillation | Halve the number of required steps in each distillation round |

PMD belongs to a different family: it **matches distributions**, not pointwise outputs, by regressing a reweighted average of student samples to the teacher's **posterior mean** under a known corruption process.

## How it works

The teacher defines the posterior $p(x_0 \mid x_t)$ induced by the forward noise process. Its *mean* $\mu(x_t) = \mathbb{E}[x_0 \mid x_t]$ is what a single teacher denoising step predicts.

For a student $G_\theta$:

1. **Draw samples** $x^{(i)} = G_\theta(z^{(i)}, c)$ for a shared condition $c$.
2. **Compute an exponentially-weighted average** of those samples: $\bar{x} = \sum_i w_i \cdot x^{(i)}$ with $w_i \propto \exp(-\lambda \cdot E(x^{(i)}))$ for some energy $E$ defined by the teacher.
3. **Match** $\bar{x}$ to the teacher posterior mean at a **shifted input** — under Gaussian corruption this input has a closed form, and the whole target collapses to a single teacher evaluation.

Loss (schematic):

$$
\mathcal{L}_{\text{PMD}} = \mathbb{E}\, \| \bar{x}(\theta) - \mu_{\text{teacher}}(x_t^{\text{shift}}) \|^2 .
$$

The authors prove the fixed point of this objective identifies the teacher's full conditional posterior, not just its mean — i.e. the student is pushed to the right *distribution* of outputs, not just the right expectation.

## Why it matters

- **Diversity + quality in one objective.** Consistency and score distillers often collapse modes. Posterior-matching keeps intra-prompt diversity because the target is a distribution, not a mode.
- **Cheap per step.** One frozen-teacher forward per training target — comparable per-step cost to consistency distillation, but a stronger objective.
- **Proven posterior identification.** Few-step distillation rarely comes with a correctness guarantee; PMD does under Gaussian corruption.
- **Plugs into existing diffusion stacks.** The teacher stays frozen; nothing about its training or scheduler changes. Only the student loss is new.

## Gotchas & tricks

- **Gaussian corruption assumption.** The input-shift trick that keeps training cheap is specific to Gaussian noise schedules. Non-Gaussian schedules (rectified flow with alternate noise) need a redone derivation.
- **Exponential weights can collapse.** If one sample dominates the softmax, the "distribution-matching" reduces to pointwise matching. The temperature on the weights is a real hyperparameter.
- **Batch of students per training step.** PMD needs multiple $x^{(i)}$ per update to form the average. Not a free lunch on GPU memory.
- **Doesn't fix a bad teacher.** The student's ceiling is the teacher's posterior; PMD gets you there, it doesn't exceed it.

## Sources

- Paper: *Posterior Matching Distillation* — 2026.
- Related: *Consistency Models* — Song et al., 2023.
- Related: *Score Distillation Sampling / Diff-Instruct* — Luo et al., 2023–2024.
