# Trainable Steering Vectors
*Depth — fitting activation-space perturbations to recover an RL post-training gain and probe its geometry.*

**TL;DR:** Instead of asking "what weights change under RLVR?", ask "what minimal activation perturbation reproduces the gain?" **Trainable steering vectors** are small learnable additive terms on the residual stream, fit end-to-end to match RLVR performance from a frozen base. The exercise exposes the *geometry* of RL gains: they live on a **low-dimensional** manifold (but not infinitely compressible), and effective directions sit in the **low-variance complement** of the activation principal subspace, not along the top PCs. Published alongside **Alpha-Stabler**, a plug-and-play RLVR training-stability monitor built from the same subspace idea.

**Prereqs:** [README.md](README.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [u-space.md](u-space.md) · [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

A diagnostic method that treats "recover an RL gain" as a bottleneck experiment on activation space. Fix the base model's weights; attach a trainable additive bias (or low-rank bias) to selected residual-stream positions; optimise it on the RL reward (or on behavioural cloning of an RL checkpoint). The learned vectors are then the *minimal* activation perturbation that reproduces the gain.

The rank, position, and principal-subspace alignment of the fit vectors become empirical measurements of what RLVR actually does to the model's representations.

## How it works

- **Attach steering vectors.** Add a trainable $\Delta \in \mathbb{R}^{d}$ (or low-rank $UV^\top$) to the residual stream at chosen layers/positions.
- **Fit on the RL objective.** Train only the steering parameters against the same reward used for RLVR (GRPO-style loss, verifiable reward), with the base model frozen.
- **Measure geometry.** PCA the fit $\Delta$s. The paper reports two empirical findings:
  - *Low-dimensional but not trivial.* A small number of dimensions suffice to recover the gain, but single-direction (rank-1) fits fall short — there's a floor.
  - *Low-variance complement.* The effective directions align with the **low**-variance tail of the activation PC spectrum, not the top PCs. Ultra-LoRA / top-PC adapters miss them.

**Alpha-Stabler.** Monitors the online projection of training activations onto this subspace during RLVR and flags training instability (e.g. collapse, drift) with a lightweight add-on.

## Why it matters

- Explains why **LoRA / steering baselines on RLVR** can look weirdly brittle: standard low-rank adapters tend to live in the top-PC subspace, which the geometric finding says is the wrong place.
- Gives a cheap tool for predicting when a given adapter rank *will* work — if the top-$k$ PCs don't cover the low-variance tail where RL lives, don't expect LoRA to match RLVR.
- Converts an interpretability finding into an engineering monitor (Alpha-Stabler) that pays for itself in RL training stability.

## Gotchas & tricks

- Fit sensitivity: optimisation hyperparameters for the steering vectors matter. Keep learning rates small and warm up — these are tiny parameter sets but still interact with the full base model.
- Position choice matters more than layer count: a few well-chosen token positions beat applying $\Delta$ everywhere.
- The low-variance-complement result is for RLVR; preference-RL (RLHF with learned RMs) may populate different subspaces.
- Don't over-generalise: the fit vectors reproduce *behaviour*, not necessarily the *reasons* for RLVR gains. Causal claims require interventions, not just fits.

## Sources

- Paper: *Learning to Steer, Steering to See: Unveiling the Geometry of RLVR in Large Language Models via Trainable Vectors* — Cai, Cao, Yin, Xu et al., Tencent Hunyuan, 2026 — [arXiv:2609.34344](https://arxiv.org/abs/2609.34344).
