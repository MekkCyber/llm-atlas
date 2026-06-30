# Gist Token Attention (SSA)
*Depth — sparse attention via interleaved summary tokens, with no architectural changes.*

**TL;DR:** Simplified Sparse Attention (SSA) trains a standard Transformer to pack each chunk's information into special **gist tokens** by continued pre-training under a restrictive attention mask. At inference, queries score chunks against the small gist set and only the top-k chunks are *selectively unfolded* back into full attention. No new modules, no auxiliary KV cache — just masking and a continued-pretraining recipe.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [mla](../architectures/mla.md), [dca](../fundamentals/dca.md)

---

## What it is

A sparse-attention scheme that avoids the architectural complexity of prior work (custom kernels, separate router networks, auxiliary KV caches). The model is the same dense Transformer; the sparsity comes entirely from how tokens are arranged and masked during continued pre-training and how chunks are selected at inference.

## How it works

**Continued pre-training.** Sequences are chopped into chunks. After each chunk, a small number of *gist tokens* are inserted. An attention mask forces gist tokens to attend to their preceding chunk, and forces subsequent tokens to read from those gists (rather than the raw chunk). Standard next-token loss; the gradient pressure pushes gist representations to become useful summaries.

**Inference (SSA).**

1. Encode the full prefix into chunks of gist tokens.
2. For the current query, score each chunk by attention against its (already-cached) **gist tokens only** — bandwidth-cheap.
3. Pick top-k chunks; *unfold* them by reintroducing their raw KV entries into the attention computation.
4. Decode with attention restricted to (gists ∪ unfolded raw KV).

**H-SSA (hierarchical).** Build gists *of* gists, giving log-linear decoding cost while preserving high-compression accuracy.

## Why it matters

- **No bespoke components.** Drops into existing dense-attention serving stacks; gist tokens are normal tokens with an unusual mask.
- **Memory bandwidth, not just FLOPs.** Scoring against gists instead of full KV is the actual win on modern hardware — the bandwidth wall is what hurts long-context decoding.
- **Sparse beats dense in RAG.** Reported >5.7 LongBench RAG-point gain over the continued-pretrained full-attention baseline; selective unfolding acts as a learned noise filter on retrieved-chunk distractors.
- **Decoding latency stays ≈flat** as context grows; up to **3.37× speedup over Flash-Decoding**.

## Gotchas & tricks

- Compression ratio is set by the gist:chunk ratio; pushing past 32× is where H-SSA's hierarchy starts to matter.
- The continued-pretraining stage is non-trivial — naive insertion of gist tokens without the right mask degrades base perplexity.
- Top-k selection is per-query, so latency wins assume the gist KV cache fits resident; gist size should be tuned to your serving budget.

## Sources

- Paper: *Simplified Sparse Attention via Gist Tokens* — arXiv:2604.20920 — https://arxiv.org/abs/2604.20920
- Code: https://github.com/yuzhenmao/simplified-sparse-attention
