# Soft Multi-Token Prediction (Soft-MTP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard [multi-token prediction](mtp.md) uses cross-entropy on each MTP head's output. That gradient decomposes *exactly* into a useful **KL self-distillation** signal (each MTP head pulled toward the model's own next-token distribution) plus a **low-SNR residual** that buries the KL signal — the misalignment grows with scale. Soft-MTP is a one-line change that keeps the KL part and drops the residual, using a straight-through gradient-routing trick on the shared linear output head. No new parameters, no measurable overhead, improves NTP perplexity, speculative-decoding acceptance, and zero-shot accuracy at every scale from 167M to 1.2B.

**Prereqs:** [mtp](mtp.md)
**Related:** [deepseek-v3](../case-studies/deepseek-v3.md)

---

## What it is

MTP trains auxiliary heads $h_{\text{MTP},k}$ on top of the backbone to predict token $t+k+1$. The standard training signal is cross-entropy against the target token for each head. Soft-MTP replaces that cross-entropy with the *KL* between the $k$-th MTP head's output and the main model's next-token-prediction (NTP) distribution for the same position — but computes the KL gradient *through the cross-entropies you were already computing*, not by materializing the full KL loss.

## How it works

Let $p = \pi_\theta(\cdot \mid \text{NTP})$ and $q = \pi_\theta(\cdot \mid \text{MTP head } k)$ share the linear LM head $W$. The authors show that

$$
\nabla_\theta \mathrm{KL}(q \,\|\, p) = \nabla_\theta \mathrm{CE}(q, y) - \nabla_\theta \mathrm{CE}(p, y) + \text{terms that vanish via straight-through routing}
$$

for any one-hot target $y$. In practice: compute both cross-entropies (which the MTP trainer already does), subtract the gradients, and route the straight-through so that only the KL-signal part flows back. The classical MTP objective includes an additional residual term that is noisy — it has low signal-to-noise and swamps the KL direction as models scale up.

**One-line implementation.** Supports both DeepSeek- and Meta-style MTP heads unchanged. Adds no parameters. No vocab-sized logit matrices (unlike minimizing KL directly), compatible with fused cross-entropy kernels.

## Why it matters

MTP is now a default pretraining ingredient (DeepSeek-V3 depth-1 MTP, Meta MTP exploration). The paper's clean decomposition explains a puzzle: *why does MTP scale less well than the raw math predicts?* Answer: the standard loss carries a noise term whose weight grows with scale. Stripping it recovers the gain that MTP was supposed to produce — and gives faster speculative decoding as a side effect because the heads are now better calibrated to the NTP distribution.

## Gotchas & tricks

- **Shared output head is required.** The gradient-routing identity relies on $p$ and $q$ going through the same linear $W$. Separate heads break it.
- **DeepSeek-style vs Meta-style MTP both work.** The decomposition does not depend on how the heads are arranged.
- **No extra memory.** Unlike direct KL minimization, which needs the full vocab × batch × seq softmax, Soft-MTP reuses the existing cross-entropy tensors.
- **Scale matters.** Below ~689M, standard MTP and Soft-MTP are close on MTP perplexity. Beyond that, the residual's SNR hurts standard MTP more, and Soft-MTP pulls ahead.
- **Speculative decoding improves.** Draft logits are now better matched to the main model, raising acceptance rate — a free secondary win.

## Sources

- Paper: *Multi-Token Prediction Done Right* — J. Teoh et al., MIT, 2026 — arXiv:2610.38331 — introduces the gradient decomposition and the Soft-MTP recipe.
- Paper: *Better & Faster Large Language Models via Multi-token Prediction* — Gloeckle et al., 2024 — the Meta-style MTP objective Soft-MTP critiques.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — depth-1 MTP in production.
- Code: https://github.com/JaydenTeoh/Soft-MTP
