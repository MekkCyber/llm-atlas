# Flow Language Model (Lafa)
*Depth — a blockwise continuous-flow architecture that keeps autoregressive serving tooling while decoding inside blocks in parallel.*

**TL;DR:** Decoder-only transformers decode left-to-right one token at a time; flow- and diffusion-LMs decode all tokens in parallel but historically don't scale and lose prefix caching. Lafa (Large Flow Language Model) splits the difference: **causal attention across blocks** (so you keep a working prefix cache) + **bidirectional flow matching within each block** (so each block decodes its tokens in parallel). Demonstrated at 8B parameters on <30B tokens, achieving ~2.54× throughput over an autoregressive baseline at comparable quality, with native support for prefix caching and self-speculative decoding.

**Prereqs:** [transformer-block](transformer-block.md), [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mtp](../pre-training/mtp.md)

---

## What it is

A continuous-latent language model trained with a flow-matching objective, organized as a sequence of fixed-size blocks. Each block of $B$ tokens is generated jointly via a short flow trajectory; blocks follow one another autoregressively.

## How it works

**Blockwise structure.** The sequence is cut into fixed-length blocks (typical $B \in \{8, 16\}$). Attention is causal *across* blocks and bidirectional *within* a block.

**Flow matching inside a block.** For block $k$, start from a Gaussian latent $z^k_0$ and solve a learned velocity field $v_\theta(z^k_s, s \mid \text{prefix})$ for $s \in [0, 1]$:

$$ \frac{dz^k_s}{ds} = v_\theta(z^k_s, s \mid \text{prefix}), \quad z^k_1 \to \text{tokens of block } k. $$

Conditioning on the prefix is through the causal attention over earlier blocks; within-block positions attend bidirectionally so the flow has access to the whole block as it refines.

**Decoding.** Blocks are produced in order. Within a block, the flow is integrated in a small number of steps (typical 4–8) with the whole block updating in parallel each step.

**Prefix caching.** Because across-block attention is causal, the KV cache for earlier blocks is reusable — same mechanism as autoregressive decoders.

**Self-speculative decoding.** The same model, run with fewer flow steps, serves as a cheap draft; a full-step run on the same block verifies. Because the draft and target share weights, there is no separate draft model to train.

## Why it matters

- **First clean 8B-scale demonstration** that a flow-/diffusion-LM can be scaled with modest training budget. Earlier non-autoregressive LMs plateaued at ≤1B scale.
- **Keeps the autoregressive serving stack.** Prefix caching + speculative decoding are the two levers vLLM/SGLang-class serving systems depend on; Lafa's design preserves both.
- **Parallel in-block decoding is a real throughput win.** Reported ~2.54× over matched AR baseline — different in character from quantization or batching speedups.

## Gotchas & tricks

- **Block boundary artifacts.** Transitions between blocks can show quality dips; the paper tunes block size against this.
- **Flow steps vs quality.** Fewer steps = faster but noisier blocks; the draft/target split in self-speculative decoding exploits this on purpose.
- **Not strictly a token model.** Continuous-latent flow + discretization at the end means some LM tooling (direct logit edits, exact token-level log-probs) needs adaptation.
- **Scaling data budget is low.** 30B tokens is well below modern frontier LM training; benchmark comparisons are to similarly-trained AR baselines, not to top open LMs.
- **Inside-block loss normalization matters.** Standard flow-matching gradients weighted uniformly in $s$ underperform time-weighted schedules at long block lengths.

## Sources

- Paper: *Large Flow Language Models* — anonymous (listed on HF Daily Papers 2026-10-21), 2026 — [arXiv:2610.28559](https://arxiv.org/abs/2610.28559).
