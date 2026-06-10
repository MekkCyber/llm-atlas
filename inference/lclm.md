# Latent Context Language Models (LCLMs)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An **encoder-decoder context compressor** for long-context LLM inference: a small encoder (~0.6B) maps a long token sequence to a much shorter sequence of latent embeddings, which a larger decoder (~4B) reads as if they were ordinary tokens. Pretrained from scratch at production scale (350B tokens) at compression ratios 1:4, 1:8, 1:16. Closes the gap with KV-cache compression on the accuracy-efficiency frontier, and uniquely enables **agent-style skim-and-expand** workflows: an agent skims the compressed latent and selectively re-expands relevant regions.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/mla.md](../architectures/mla.md)

---

## What it is

Three families of long-context compression:

| Family | What's compressed | When |
| --- | --- | --- |
| KV-cache eviction / quantization | Per-token KV pairs, after the prefill | At inference, per-request |
| Sparse / retrieval attention | Which KV pairs are attended to | At inference, per-decode step |
| **Encoder-decoder compression (LCLM)** | The input tokens themselves, into latents | Once, before decoding starts |

LCLMs sit in the third family. The pitch: KV-cache compression operates *after* you've paid prefill cost on every token; encoder-decoder compression pays a smaller encoder forward pass and then decodes against a shorter sequence. Past work in this family lost to KV-cache methods on accuracy. LCLMs show the gap closes when you spend the pretraining budget.

## How it works

- **Architecture search first.** Pretrain many encoder-decoder variants from scratch at small scale; identify which design choices matter (encoder depth, projector type, latent ratio, where the decoder reads). The paper's chosen design is the variant that emerges from this search.
- **Joint pretraining.** Encoder and decoder are pretrained together as a single language model on 350B tokens each, at the chosen compression ratio. The encoder is *not* a frozen retriever bolted onto a pretrained LM — the decoder learns to read the encoder's latent format from scratch.
- **Multiple ratios.** Three model families at 1:4, 1:8, 1:16 compression, so the practitioner can pick the right point on the Pareto frontier.

At inference: tokenize the long prompt → encoder produces a latent of length $\lceil L / r \rceil$ → decoder generates autoregressively, attending to the latents as if they were ordinary KV. The decoder's KV cache only ever holds the compressed latents plus the generated tokens.

## Why it matters

- **Pareto improvement** across general-task accuracy, compression speed, and peak memory vs prior encoder-decoder compressors and several KV-cache methods. The frontier is the comparison; LCLMs push it out.
- **Agent skim-and-expand** is unique to this family. With a compressed latent, an agent can quickly scan a million-token context, then re-encode a specific region at full resolution when needed. KV-cache compression can't do this — once you've quantized or evicted, you can't recover the original signal.
- **Production inference engines** can host the decoder without modification — it's just a Transformer that happens to see latents. The encoder is a one-time preprocessor, easy to disaggregate.

## Gotchas & tricks

- **Compression ratio is fixed at pretraining.** You don't get to dial it per request without retraining. The paper ships three families; expect to pick at deployment.
- **Re-expansion needs the original tokens.** "Skim and expand" assumes you've kept the original input around (on cheaper storage) to re-feed the encoder. Otherwise expansion is impossible.
- **Generation-time KV is still standard.** The compression is on the *prompt*; the decoder's own generated tokens still have full KV. Long generations don't help LCLMs as much as long prompts do.
- **Drift on adversarial prompts.** Compressed latents lose fine-grained detail; tasks that need exact token-level recall (needle-in-haystack, exact-quote retrieval) hurt at high compression. The paper shows 1:4 is safe for most tasks; 1:16 trades quality for memory.
- **Encoder is non-trivial.** 0.6B is small relative to the decoder but not free — at very short prompts it's pure overhead. LCLMs win in the long-prompt regime they were trained for, not at chat-length inputs.

## Sources

- Paper: *End-to-End Context Compression at Scale* — Li, McLeish, Chen, Kalra, … Lotfi, Goldblum, Izmailov — NYU / Modal Labs / UMD / Princeton / Columbia / Harvard / LLNL / FAIR at Meta, 2026 — arXiv 2606.09659.
