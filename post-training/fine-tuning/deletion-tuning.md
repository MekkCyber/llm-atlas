# Deletion Tuning
*Depth — fine-tune by learning a binary mask that zeroes a tiny fraction of pretrained weights.*

**TL;DR:** The only trained parameter is a binary mask over the gate projection of each transformer block. Multiplying the frozen pretrained weights by the mask ablates fewer than **0.05%** of them — and that alone matches full fine-tuning on RLVR, SFT, and on-policy distillation, while forgetting much less of the base model. The checkpoint is hundreds-to-thousands of times smaller than the full-FT delta.

**Prereqs:** [README.md](./README.md) (fine-tuning overview), [../rlvr.md](../rlvr.md) (one of the three regimes it's tested in).
**Related:** [../grpo.md](../grpo.md), [../_post-training.md](../_post-training.md), [../../interpretability/README.md](../../interpretability/README.md) (circuits intuition).

---

## What it is

A fine-tuning method that *adds* no parameters and *updates* no weights. For each transformer block, attach a learnable binary mask $m \in \{0, 1\}^{d}$ over the gate-projection weight matrix $W_{\text{gate}}$ (or equivalently, over its columns). Forward pass uses $W_{\text{gate}} \odot m$; other projections stay exactly as pretrained. Only $m$ is trained — and once optimized, the model's "fine-tuned" state is fully described by which bits of $m$ are 0.

## How it works

Train the mask through the downstream objective:

- **RLVR**: the standard GRPO policy loss, backpropagating through the straight-through estimator for $m$.
- **SFT**: standard token-level cross-entropy, same trick.
- **On-policy distillation**: student = model-with-mask, teacher = fixed teacher; KL loss on token distributions.

The mask starts at all-ones (no deletion). Gradients push the continuous surrogate down for weights that interfere with the target behavior; a threshold at the end snaps to {0, 1}. Fewer than 0.05% of weights end up at 0.

Why the *gate* projection specifically: in gated FFNs (SwiGLU / GeGLU), the gate is the multiplicative router that decides which hidden dimensions fire. Masking a gate column effectively deactivates one computational pathway rather than perturbing a feature — a much more "clean" edit than masking down- or up-projection weights.

## Why it matters

- **New capabilities as path-pruning rather than parameter-writing.** The paper's results suggest that pretraining already contains the needed computations; post-training mostly suppresses competitors. That inverts the usual "fine-tuning adds skill" story.
- **Tiny deltas.** A full-FT checkpoint is >100× bigger than a mask. Deploying N task adapters becomes nearly free.
- **Retention.** Under RLVR, Deletion Tuning holds prior-task accuracy near the base model while full FT, LoRA, and random sparse FT each lose up to 5.4 points — a direct continual-learning advantage.
- **Analytic fingerprint close to full-FT.** By stable rank, spectrum drift, and principal-weight overlap, Deletion Tuning's post-training weight statistics are *closer to full fine-tuning* than LoRA's are — evidence that masking is finding real task geometry, not a low-rank surrogate.

## Gotchas & tricks

- **Straight-through estimator is standard**; the mask is a sigmoid during backprop and binarized at inference. Temperature annealing stabilizes the threshold crossing.
- **Not architecture-agnostic.** Requires a gated FFN. For pure MLP stacks (older architectures) you need to pick a different projection, and the paper's quality results may not transfer.
- **Random sparse FT is a weaker baseline** than randomly masking the same fraction of weights would suggest — the paper shows learned masks win by a wide margin, so the choice of *which* 0.05% matters.
- **Masks don't compose by OR.** Stacking two task masks is not the same as training on both tasks jointly; expect interference, especially if the task frontier overlaps.

## Sources

- Paper: *The Surprising Effectiveness of Deleting Weights in LLM Post-Training* — Lu, Yang, Lasby, Pote, Ioannou, Ren, Agentic Learning AI Lab, 2026 — [arXiv:2611.10866](https://arxiv.org/abs/2611.10866).
