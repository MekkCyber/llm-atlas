# Visual Routing (Certificate-Guided)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-free decoding method for diffusion vision-language models. At each denoising step, run two forward passes — one with image tokens, one without — and if their predictions agree on the just-decoded span, treat the agreement as a *certificate* that the image was not needed and skip image conditioning for a window of future steps. Introduced in LOWN; keeps 99.6% of LLaDA-V's score at 35.3% of decoding FLOPs across 10 benchmarks.

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [../pre-training/mtp](../pre-training/mtp.md)

---

## What it is

Diffusion VLMs pay a visual-tower cost at *every* denoising step. In practice most steps don't actually need to re-read the image — the token being decoded is determined by prior text context alone. Visual routing is the technique of deciding, per step and *without any extra model*, whether to pay that cost.

The "certificate" is the key idea: a cheap test that proves the image wasn't needed, implemented as prediction agreement between an image-free and image-conditioned forward pass.

## How it works

At decoding step $t$:

1. Run a **text-only** forward pass → prediction $\hat y^{text}_t$.
2. Run the **image-conditioned** forward pass → prediction $\hat y^{img}_t$.
3. If $\hat y^{text}_t == \hat y^{img}_t$ on the just-decoded span (or within some tolerance for logits), issue a certificate: the model agrees without seeing the image.
4. Under a certificate, skip the image-conditioned forward for the next $k$ steps and emit text-only predictions.
5. On any disagreement during a skipped window, re-enable image conditioning immediately.

Nothing is learned. Nothing is cached beyond the usual KV cache. The routing is purely a stability test on the model's own predictions.

## Why it matters

- Diffusion VLMs scale decoding compute with step count; cutting ~65% of image-conditioning passes directly reduces latency and energy per generated token.
- Training-free: no new weights, no distillation, no retraining. Applies to any released diffusion VLM.
- Reframes "where should the model look at the image" as an inference-time routing problem rather than an architecture question. Similar certificates can be defined for autoregressive VLMs by comparing an image-ablated prefill against the normal one.

## Gotchas & tricks

- The two forward passes aren't fully overlapped: text-only is cheaper but still non-zero. Overall speedup is dominated by how often certificates fire; on image-heavy benchmarks (DocVQA) the fraction drops.
- Choosing the skip window $k$ is the main hyperparameter. Too long and you miss the moment the image becomes relevant again; too short and you lose the gain. The paper's default is small and conservative.
- The agreement test is sensitive to token-granularity decisions in diffusion decoding; sliding-window masked decoding (LLaDA-V default) is well-behaved, strictly block-parallel decoding less so.
- Does not stack with image-token pruning methods that remove tokens *before* the tower — those are orthogonal savings.

## Sources

- Paper: *Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs (LOWN)* — Cho, Kim, Shin, Yang, KAIST AI, 2026 — [arXiv:2610.36671](https://arxiv.org/abs/2610.36671).
- Related: *LLaDA-V* — the diffusion VLM evaluated in the paper.
