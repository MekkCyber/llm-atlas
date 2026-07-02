# Diffusion-Based Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Speculative decoding accelerates LLM inference by having a cheap *draft model* propose candidate tokens that the target model verifies in parallel. Diffusion-based speculative decoding replaces the AR draft with a **block-diffusion** draft that generates many tokens per forward pass through denoising, achieving state-of-the-art throughput on decode-heavy workloads. BlockPilot (Zhang et al., 2026) adds an instance-adaptive policy that predicts the optimal block size per prompt, adding another 20–30% speedup on top.

**Prereqs:** [README](README.md)
**Related:** [block-diffusion-lm](block-diffusion-lm.md) · [../systems/partial-rollouts](../systems/partial-rollouts.md) · [../pre-training/mtp](../pre-training/mtp.md)

---

## What it is

Classical speculative decoding uses a small AR draft model $q$ and a large target $p$:

1. Draft samples $k$ tokens autoregressively from $q$.
2. Target scores those $k$ tokens in one forward pass.
3. Rejection-sample: for each drafted token, accept with probability $\min(1, p/q)$; on rejection, resample from the residual distribution $(p - q)_+$ and stop.

Draft speed × acceptance rate = throughput. The bottleneck is that the draft is *autoregressive*: even though the target verifies in parallel, the draft still spends $k$ sequential forward passes to produce $k$ candidate tokens.

**Diffusion-based speculative decoding** replaces the AR draft with a **block-diffusion LM** that produces $k$ tokens per forward pass through parallel denoising. Because the draft is now parallel, wall-clock speedups can exceed those of AR-draft speculative decoding for the same acceptance rate.

## How it works

### Block-diffusion draft

A block-diffusion LM (see [block-diffusion-lm](block-diffusion-lm.md)) generates fixed-size blocks of tokens by iterative denoising. In a few denoising steps, the draft produces a full block of size $B$. Choosing $B$ well is the whole game:

- **Small $B$** → few tokens per forward → low throughput.
- **Large $B$** → many tokens per forward but higher rejection risk on later tokens in the block.

### BlockPilot: instance-adaptive block size

BlockPilot observes that the *optimal* $B$ varies substantially across prompts, and that the useful range is small and concentrated near the training block size. Cast block-size selection as a **lightweight policy learning** problem:

1. Extract the prefill hidden state after the prompt encodes.
2. A small policy head predicts $B^\star$ from that hidden state.
3. Decode using $B^\star$ for the rest of the response.

The policy prediction happens *once*, after prefill. Overhead is negligible; the prediction plugs into any block-diffusion draft.

### Verification

Target verification is standard: for each draft block, the target scores all tokens in parallel, accept-reject up to the first rejection, and continue. Because block-diffusion drafts are parallel, wall-clock speedup depends only on target verification cost per accepted token.

## Why it matters

- **Higher throughput than AR-draft SD.** BlockPilot reports **acceptance length 5.92** and **4.20× speedup** on Qwen3-4B at $T=1$, plug-and-play across diffusion-draft backends.
- **Serves large-model latency budgets.** Speculative decoding is the standard cheap knob for LLM serving; diffusion drafts push the throughput ceiling further.
- **Per-instance adaptation is cheap.** BlockPilot's policy head is tiny; it's the "always adapt" version of the "always fix $B$" default.

## Gotchas & tricks

- **Draft-target vocabulary match.** Draft and target must share tokenizer; verification math breaks otherwise.
- **Rejection cascade.** Once a token in the block is rejected, all tokens after it must be re-generated. Large blocks amplify this cost — the whole point of BlockPilot is to size $B$ so rejections happen late.
- **Temperature interacts with acceptance.** High $T$ on the target lowers per-token match probability; adjust $B$ or the draft temperature accordingly. BlockPilot experiments were at $T=1$.
- **Cache reuse.** Both draft and target benefit from KV-cache reuse across draft blocks; block-diffusion drafts must expose their KV cache to the verifier.

## Sources

- Paper: *BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based Speculative Decoding* — Zhang, Hu, Wang, Mo, Xiao, Chu, 2026 — Alibaba AMAP; per-instance block-size policy.
- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias, 2022 — classical AR speculative decoding.
- Paper: *Block Diffusion Language Models* (2025 lineage) — the block-diffusion LM substrate; see [block-diffusion-lm](block-diffusion-lm.md).
