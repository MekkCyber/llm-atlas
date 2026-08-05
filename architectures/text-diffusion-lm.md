# Text Diffusion Language Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Generate text by iteratively refining a fully- or partially-masked block of tokens via a small number of denoising steps, instead of autoregressively decoding one token at a time. Each forward pass can commit many tokens in parallel. Historically underperformed AR at matched compute, but adapting a pretrained AR model (rather than training from scratch) closes the gap and unlocks large throughput wins — DiffusionGemma is the first frontier-lab open-weight demonstration.

**Prereqs:** [multi-head-attention](multi-head-attention.md)
**Related:** [_moe](_moe.md), [../pre-training/mtp.md](../pre-training/mtp.md), [../inference/README.md](../inference/README.md)

---

## What it is

Two decoding paradigms for a transformer LM:

- **Autoregressive (AR).** Left-to-right, causal attention, one committed token per forward pass. Standard.
- **Text diffusion.** Bidirectional (within a block), commit many tokens per pass via iterative denoising. Trade some quality per FLOP for a large parallelism win.

Text diffusion is the *discrete* analogue of image diffusion. It operates on a sequence of token indices with `<mask>` as the noise state, and iteratively predicts the un-masked distribution using bidirectional attention.

## How it works

**Training objective (masked denoising).**

Sample a masking ratio $r \sim U[0,1]$ for a block of $B$ tokens. Replace $r \cdot B$ tokens with `<mask>`, train the model to predict the originals with bidirectional attention:

$$
\mathcal{L} = -\mathbb{E}_{r, \text{mask}} \sum_{i \in \text{masked}} \log p_\theta(x_i \mid x_{\text{visible}}, r)
$$

Weighting the loss by $r$ (or a related schedule) emphasises the harder, higher-noise cases.

**Inference (block-wise iterative refinement).**

```
for each block of B tokens:
    initialise all B positions as <mask>
    for step = 1 to T:
        run model forward → distribution over each masked position
        commit the k most-confident positions (top-k selection)
        keep the rest masked
    proceed to next block, conditioning on the previously-decoded blocks
```

**Block size vs full-sequence.** Two variants:

- **Full-sequence diffusion** (SEDD, MDLM): the whole target is diffused at once. Best parallelism, but loses causal streaming and forces the model to condition on both past and *future* tokens even during training.
- **Block diffusion** (DiffusionGemma-style): the sequence is split into blocks; blocks are left-to-right, positions within a block are bidirectional. Recovers streaming and left-context caching; still gets in-block parallelism.

Frontier open-weight releases favour block diffusion.

**AR → diffusion adaptation.** Rather than train from scratch, initialise from a pretrained AR checkpoint and fine-tune with the diffusion objective. The transformer architecture is unchanged; only the attention mask (bidirectional within blocks) and the loss change. This is what makes text diffusion economically viable — reuse of the AR pretraining sunk cost.

## Why it matters

- **Parallel token commit.** Each forward pass produces multiple tokens; effective ~20 tokens/step is achievable at frontier scale (DiffusionGemma).
- **Throughput.** ~1,500 tokens/sec on one H100 — order-of-magnitude better than AR of the same size even with state-of-the-art speculative decoding.
- **Hybrid decoding available.** A diffusion-trained checkpoint can also decode AR-style with small quality loss, so you can mix modes per-token or per-request.
- **KV-cache is not obsolete.** Left-context of prior blocks is reused across blocks; only the current block is bidirectional.

## Gotchas & tricks

- **Number of denoising steps matters.** Too few → poor quality; too many → throughput collapses. Sampler distillation (see [sampler-distillation](../post-training/sampler-distillation.md)) is what makes ~13 steps per 256-token block enough.
- **Confidence-based selection > random.** Committing the most-confident positions per step (top-k by predicted probability) beats random-order committing at matched step count.
- **Long-context is not free.** Bidirectional attention *within* a block still costs $O(B^2)$; keep blocks moderate (128–512) rather than sequence-length.
- **Loss weighting is fragile.** Uniform weighting over $r$ under-trains the high-mask regime that matters at inference (fully-masked block).
- **Streaming latency.** Even with block diffusion, first-token latency is `~T × forward-pass time`, not `1 × forward-pass time`. AR wins on first-token latency; diffusion wins on total-token throughput.
- **Do not evaluate against AR at fixed FLOPs.** The point is parallelism — evaluate against AR at *fixed wall-clock time* on target hardware.

## Sources

- Paper: *DiffusionGemma Technical Report* — DeepMind, arXiv:2608.00146, 2026 — the first frontier-lab open-weight text-diffusion LM.
- Paper: *SEDD: Score Entropy Discrete Diffusion* — Lou, Meng, Ermon, 2023 — foundational discrete-diffusion recipe for text.
- Paper: *MDLM: Simple and Effective Masked Diffusion Language Models* — 2024 — the masked-diffusion loss commonly used.
- Paper: *LLaDA / Block Diffusion Language Models* — 2024 — block-diffusion variants closer to DiffusionGemma.
