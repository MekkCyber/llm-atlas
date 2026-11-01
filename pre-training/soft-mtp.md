# Soft Multi-Token Prediction (Soft-MTP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The standard Multi-Token Prediction (MTP) auxiliary loss has a gradient that decomposes into a KL self-distillation signal (aligning the *k*-step head with the next-token distribution) plus a *residual* term that interferes with next-token learning. **Soft-MTP** keeps only the KL part by training each MTP head against the main head's softmax output instead of hard labels. One-line code change, no new parameters, no overhead; across 167M–1.2B models it improves perplexity, zero-shot accuracy, math/code fine-tuning, and speculative-decoding acceptance.

**Prereqs:** [mtp.md](mtp.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md), [../case-studies/deepseek-v3.md](../case-studies/deepseek-v3.md)

---

## What it is

MTP (DeepSeek-V3's depth-1 variant and Gloeckle et al.'s depth-$D$ original) attaches auxiliary heads that predict tokens *k* steps ahead and trains each with cross-entropy against the ground-truth $x_{t+k+1}$. Soft-MTP replaces that cross-entropy with a KL divergence against the frozen main head's softmax, so the auxiliary signal is "match the main head's view of the future" rather than "fit the hard token."

## How it works

For the depth-$k$ head with logits $z^{(k)}_t$ and the main head's logits $z^{(0)}_{t+k}$, standard MTP uses:

$$
L^{(k)}_{\text{std}} = -\log\,\mathrm{softmax}(z^{(k)}_t)_{\,x_{t+k+1}}
$$

Soft-MTP replaces it with:

$$
L^{(k)}_{\text{soft}} = \mathrm{KL}\!\left(\mathrm{sg}[\mathrm{softmax}(z^{(0)}_{t+k})]\;\|\;\mathrm{softmax}(z^{(k)}_t)\right)
$$

where $\mathrm{sg}[\cdot]$ is a stop-gradient. The paper's gradient derivation shows the standard loss splits as (a) exactly this KL term plus (b) a residual that pulls $z^{(0)}$ away from the next-token optimum in directions orthogonal to the KL. Dropping (b) is strictly better: the KL term alone forces the backbone to represent two-token-ahead structure *and* keeps the next-token objective fully aligned.

The MTP module and parameter-sharing scheme stay identical to depth-1 MTP; only the loss changes. Implementation is one line: swap `F.cross_entropy(logits_k, labels)` for `F.kl_div(logits_k.log_softmax(-1), main_logits.detach().softmax(-1))`.

## Why it matters

MTP is now standard in frontier pretraining recipes for both representation lift and built-in speculative-decoding drafts. Soft-MTP strictly improves on it along both axes without touching architecture or parameter counts. The gradient-decomposition argument is also a general template: *inspect whether an auxiliary loss's gradient is pulling against the primary objective, and switch to a KL-only variant if so.*

## Gotchas & tricks

- **Stop-gradient on the main head is essential.** Letting gradient flow back from the KL into $z^{(0)}$ turns Soft-MTP into mutual distillation; the paper's gain comes from one-way distillation.
- **Keep the same $\lambda$ schedule as DeepSeek-V3** (0.3 early, 0.1 late). Soft-MTP changes the loss, not the mixing coefficient.
- **Speculative-decoding acceptance goes up.** The MTP head is better calibrated to the main head's softmax after Soft-MTP, which is exactly what the verifier compares against.
- **Not the same as deep-supervision or ordinary knowledge distillation.** The teacher here is the same model's *main head on a shifted position*, not an external model.

## Sources

- Paper: *Multi-Token Prediction Done Right* — MIT, 2026 — [arXiv:2610.38331](https://arxiv.org/abs/2610.38331).
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — the depth-1 MTP baseline.
- Paper: *Better & Faster Large Language Models via Multi-token Prediction* — Gloeckle et al., 2024 — the original MTP objective.
