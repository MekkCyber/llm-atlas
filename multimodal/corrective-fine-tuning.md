# Corrective Fine-tuning (CFT)
*Depth — remap failed diffusion samples onto the model's own successful continuations to get a stable, policy-gradient-free RL-style signal.*

**TL;DR:** Video and image diffusion models are hard to RL-tune: policy gradients are noisy in continuous noise space, and reward models are expensive on video. Corrective Fine-tuning (CFT), introduced by ViDecide (2026), sidesteps RL entirely: when the model fails on a planning decision, take the noise it generated for that failure and *remap* it onto the noise distribution of the model's own successful continuations from similar prior states. Fine-tune on these remapped pairs. On multi-step planning benchmarks, ViDecide reaches 83.8% strict success vs. 23.3% for standard SFT and 67.8% for the strongest prior method.

**Prereqs:** [README](README.md), [../post-training/rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [../post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A fine-tuning recipe that turns failures into training targets without a policy gradient or a learned reward. It assumes the model generates in a diffusion / flow-matching framework (i.e., samples from a learned noise distribution), and that successful rollouts from similar states are already available in the training mixture or self-generated.

## How it works

For a model that generates video / image sequences as a chain of decisions (ViDecide's framing — see [../post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md) for the text-side analogue):

1. Run the model on the current training mix with a decision-centric schedule — each denoising step is a decision with an explicit stopping criterion.
2. Label each generated sample's success per decision (e.g., physical/logical consistency check, planning oracle, puzzle solver).
3. For each **failed decision** $d_f$, retrieve one of the model's own **successful decisions** $d_s$ from a similar state (same puzzle layout, same prior sequence).
4. Compute the noise that would have produced $d_s$ from the same prefix (the diffusion inversion trajectory) and use it as the target noise for $d_f$'s input.
5. Fine-tune with a reconstruction / flow-matching loss that pulls the failed sample's noise toward the successful one.

Because the target comes from the model's own distribution, the update is stable and doesn't require reward-model calls or REINFORCE-style variance.

## Why it matters

Three concrete properties:

- **Policy-gradient-free.** No REINFORCE variance, no reward-model inference cost, no stability tricks like PPO clipping.
- **No external reward model.** The success label is a cheap check (oracle, verifier, physical-consistency rule), not a learned RM.
- **Target is self-consistent.** Remapping onto the model's own successful outputs keeps updates in the base distribution, avoiding the drift that off-policy distillation can cause.

Reports an average strict success of 83.8% across abstract games, planning tasks, and a new multi-step robot planning benchmark, far above standard SFT (23.3%) and the strongest prior RL-style method (67.8%). Generalizes to larger puzzles and unseen layouts.

## Gotchas & tricks

- **Need a cheap success check.** CFT only works where failure labeling is easy (planning oracles, logic verifiers). For open-ended generation it reduces to self-distillation from best-of-N.
- **Similarity matching for retrieval matters.** Mapping a failure to the wrong "successful" continuation injects noise. The paper's experiments pair same-prefix same-layout retrievals.
- **Decision-centric training schedule is a prerequisite.** CFT relies on the model producing stopping decisions, not fixed-length clips; retrofitting onto a vanilla diffusion model requires that schedule change first.
- **Transfers to text?** Conceptually, same mechanism should work for text diffusion / flow matching with a verifier — unexplored in the paper.

## Sources

- Paper: ViDecide: Teaching Video Diffusion Models to Reason and Plan Step by Step — Zhang et al., 2026 — [arXiv:2611.26389](https://arxiv.org/abs/2611.26389)
- Project page: https://kevin-thu.github.io/ViDecide
