# Trainable Steering Vectors for RLVR Geometry
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use trainable activation-space vectors as a *probe* for where the gains from Reinforcement Learning with Verifiable Rewards (RLVR) live inside an LLM. Across trained models, the RL-induced delta concentrates in a low-dimensional activation subspace — but one that cannot be compressed indefinitely, and that lies mostly in the *low-variance* complement of the model's principal activation subspace. Exploiting the geometry, the paper's **Alpha-Stabler** removes the principal-subspace component of activation gradients during RL training, extending stable training to 2,000 steps and raising held-out gains.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [u-space.md](u-space.md) · [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

Standard RLVR is treated as an opaque optimization: it works, but we lack a geometric account of where its gains land in the network. This paper injects trainable steering vectors at a chosen layer and asks: *how much of the RL-trained model's advantage over the base can be recovered by learning the right small vector?* The answer places an upper bound on the "effective dimension" of RLVR updates and reveals which activation subspace carries them.

## How it works

- **Steering-vector probe.** Attach a trainable vector to the residual stream at a chosen layer; freeze everything else; train only this vector to recover RL gains. The vector's dimension can be constrained (rank-$r$ projection) to measure how low-dimensional the gains are.
- **Geometric decomposition.** Decompose activations into a principal subspace (top variance directions) and its complement. Measure the gain carried by each component.
- **Finding.** The useful directions mostly lie in the *low-variance complement* of the principal subspace. Compressing beyond a modest rank hurts; the subspace is small but non-trivial.
- **Alpha-Stabler.** During actual RL training (GRPO-style), monitor intrusion of the policy gradient into the principal activation subspace and project it out — "stabilize" the update by removing the component along the dominant variance directions.

## Why it matters

- Gives interpretability a concrete lever on RL training: a geometric claim ("RL updates live in the low-variance complement") turns into a stabilizer that extends usable training by an order of magnitude.
- Reframes the common "clip harder / tune KL" story: training instability can be a geometry problem (gradient leaking into principal directions), not just a magnitude problem.
- Shows that *small-but-not-tiny* subspaces carry the RL contribution; a sweet spot for LoRA-style parameter-efficient RL finetuning.

## Gotchas & tricks

- Alpha-Stabler needs a reliable estimate of the principal subspace; a running SVD on activations is cheap but drifts under heavy policy change.
- Low-rank steering is a *measurement*, not necessarily a deployment path: training a shippable model via a vector alone is harder than reading the vector as a probe.
- The claim that RL gains live in the low-variance complement is model- and task-specific; expect the ratio and location to shift across backbones.

## Sources

- Paper: *Learning to Steer, Steering to See: Unveiling the Geometry of RLVR in Large Language Models via Trainable Vectors* — Cai, Cao, Yin et al., Tencent Hunyuan, 2026 — [arXiv:2609.34344](https://arxiv.org/abs/2609.34344).
