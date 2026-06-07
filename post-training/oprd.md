# OPRD — On-Policy Representation Distillation
*Depth — distil a reasoning teacher into a student at the hidden-state level on the student's own rollout trajectories.*

**TL;DR:** Standard knowledge distillation matches the *output distributions* of teacher and student. For reasoning models, that signal is weak: rewards are sparse, sampling variance dominates late-stage training, and the LM-head projection throws away most of the information the teacher's hidden states carry. **OPRD** (Yang et al., 2026) aligns student and teacher **hidden states** at selected layers, *on the trajectories the student samples*. Same rollouts, denser supervision, no extra compute — 1.44× faster training and up to 54% lower memory than competing methods on AIME-class math benchmarks.

**Prereqs:** [post-training/_rl.md](./_rl.md), [post-training/reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md)
**Related:** [post-training/grpo.md](./grpo.md), [evaluation/aime.md](../evaluation/aime.md)

---

## What it is

Reasoning distillation has been the dominant export pipeline for frontier reasoning models: a big teacher (R1-class) is distilled into a small student (1B–32B) that retains most of the math/code capability at a fraction of the inference cost. The standard recipe is output-level KL with on-policy rollouts.

OPRD swaps the loss surface:

$$
L_{\text{OPRD}} = \sum_{\ell \in \mathcal{S}} \, \mathbb{E}_{\tau \sim \pi_{\text{student}}} \, \big\| h^{\text{student}}_\ell(\tau) - h^{\text{teacher}}_\ell(\tau) \big\|^2
$$

where $\mathcal{S}$ is a subset of layers chosen for alignment, $h_\ell$ is the hidden state at layer $\ell$, and $\tau$ is sampled from the student. The teacher only runs *forward* over $\tau$ — no additional rollouts.

## How it works

1. **Pick alignment layers.** A handful of layers (often early-mid and late) are chosen; aligning every layer is wasteful and can overconstrain.
2. **Roll out from the student.** Sample a batch of trajectories with the student policy — exactly the same rollouts an on-policy RL trainer would do.
3. **Teacher forward pass.** Run the teacher forward over those trajectories (no sampling, just a forward pass) to extract its hidden states.
4. **Hidden-state regression.** Compute MSE between student and teacher hidden states at the chosen layers; backprop through the student.
5. **Compose with RL.** OPRD slots cleanly into a GRPO loop: keep the verifiable-reward update for outcome shaping and add the representation loss for dense supervision.

### Why same-trajectory supervision matters

Output-space distillation samples $\tau$ from the *teacher* and asks the student to match teacher logits. The student never sees losses on its own mistakes. OPRD samples $\tau$ from the *student* and supervises the student's representation of its own trajectories — the loss is computed exactly where the student is and where it will be next step. On-policy + hidden-state = dense + correctly localised.

## Why it matters

- **Removes the LM-head projection bottleneck.** Hidden states carry orders of magnitude more information than logits over a 100K-token vocab; matching them is a much richer signal.
- **No extra rollouts.** The teacher only runs forward over student rollouts. Compared to teacher-policy distillation, that halves the rollout budget.
- **Compounds with RLVR.** Acts as a regulariser on the student's representation while RLVR pushes the policy.
- **Concrete savings.** 1.44× faster training, 54% less memory on AIME 2024 / 2025 / AIMO, with the student matching teacher performance.

## Gotchas & tricks

- **Layer choice is a hyperparameter.** Too few: weak signal. Too many: overconstrains, hurts capability. 3–5 chosen layers is a typical sweet spot.
- **Hidden-dim mismatch.** If teacher and student have different widths, you need a learned projection at each alignment layer. Initialise to identity-like and decay regularise.
- **Loss scale vs RL reward.** MSE and policy gradient live on different scales; balancing the two objectives is the engineering work. A warmup schedule on the OPRD term avoids early-stage RL collapse.
- **Not a replacement for a strong base.** Distillation can only transfer behaviour the student is architecturally capable of representing; very narrow students still saturate.
- **Different from feature distillation in vision.** Older feature-distillation work matches activations on the teacher's data; OPRD's novelty is doing it on-policy in the student's own trajectory distribution.

## Sources

- Paper: *OPRD: On-Policy Representation Distillation* — Yang, Zhu, Song, Wang, Xia, Zheng, Ma, Chen, Wang, Chen (Zhejiang University), 2026 — [arXiv:2606.06021](https://arxiv.org/abs/2606.06021).
