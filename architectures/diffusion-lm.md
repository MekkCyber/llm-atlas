# Diffusion Language Model (DLM)
*Depth — a language model that generates text by iterative mask-fill denoising rather than autoregressive next-token sampling.*

**TL;DR:** A diffusion language model (DLM) replaces autoregressive left-to-right decoding with iterative denoising over token sequences. The input is a fully-masked (or noised) sequence and the model fills a subset of positions on each step, parallelizing generation across the sequence. LLaDA (2024) and DiffuLLaMA (2024) are the first competitive 7B-scale DLMs; BYOD (2026) shows AR checkpoints can be converted into DLMs with LoRA adapters only. The appeal is **parallel decoding** and bidirectional context during generation; the cost is a larger per-step compute budget and a sampler-step budget to match AR quality.

**Prereqs:** [transformer-block.md](./transformer-block.md) · [multi-head-attention.md](./multi-head-attention.md)
**Related:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md) · [../fundamentals/bpe.md](../fundamentals/bpe.md)

---

## What it is

A DLM models `p(x)` as the stationary distribution of an iterative denoising process over discrete token sequences. The standard forward process is **absorbing masking**: with increasing noise level, more token positions are replaced by a special `[MASK]`. The reverse process (the model) takes a partially masked sequence and predicts masked positions in parallel. Sampling is a schedule of denoising steps `T → 0` that progressively unmasks positions until the sequence is clean.

Compared to an autoregressive model:

| | Autoregressive | DLM |
|---|---|---|
| Decoding | strictly left-to-right | parallel over masked positions |
| Attention during generation | causal | bidirectional |
| Sampler budget | `N` forward passes for `N` tokens | `k` denoising steps regardless of length (`k ≪ N` possible) |
| Loss | next-token cross-entropy | masked-token cross-entropy (BERT-style, but with random masking level) |

## How it works

**Training.** Sample a sequence `x` and a masking fraction `t ∈ (0, 1]`. Mask that fraction of positions uniformly at random. The model receives the masked sequence and is trained to predict the original tokens at masked positions with cross-entropy, weighted by `1/t` (so sparse-masked samples contribute less).

**Sampling.** Start from a fully masked sequence of the desired length. For step `t = T, T−1, …, 1`: the model predicts a distribution over all masked positions. A **remasking schedule** decides how many positions to unmask this step and which ones (lowest-entropy first, or random). Fill those; mask the rest; continue.

**BYOD conversion.** Freeze the AR backbone. Attach LoRA adapters that absorb the mask embedding, the attention-mask flip (causal → bidirectional), and a small denoising head. Train with the masked-cross-entropy objective. BYOD reports this works on Gemma 2 9B, Llama 3.1 8B, Qwen 2.5 7B, Ministral 8B; quality at low step budgets is the main remaining gap versus natively-trained DLMs.

## Why it matters

DLMs decouple **sequence length** from **sampler steps**. If `k` steps suffice to produce `N` tokens, generation is `N/k×` faster than AR. They also natively support **constrained generation** (any subset of positions can be fixed) and **non-left-to-right** prefills — attractive for editing, infilling, and reasoning that revisits earlier tokens. LLaDA 8B reaches instruction-tuned AR parity on several benchmarks, which is the first strong empirical signal that DLMs scale.

## Gotchas & tricks

- **Step budget is the whole tradeoff.** At `k = N` a DLM is strictly slower than AR; wins only kick in for `k ≪ N`.
- **Output quality at low `k` is still the blocker** for BYOD-converted models — the LoRA path hasn't fully learned the distribution's few-step denoisability.
- **Attention is bidirectional at inference.** KV-caching needs a different design than AR — many standard inference stacks (vLLM) assume causal attention.
- **Remasking schedule matters.** Confidence-based (lowest-entropy-first) typically beats uniform random; aggressive schedules save steps at a quality cost.

## Sources

- Paper: *LLaDA: Large Language Diffusion Models* — Nie et al., 2024 — first 8B-scale discrete DLM at instruction-tuned AR parity.
- Paper: *DiffuLLaMA* — 2024 — AR-to-DLM conversion with full-weight finetuning.
- Paper: *BYOD: Build Your Own DLM — Efficient Autoregressive-to-Diffusion Language Model Conversion* — Kuiper and Bagheri, UMC Utrecht, 2026 — [arXiv:2610.41574](https://arxiv.org/abs/2610.41574) — the LoRA-only conversion recipe.
