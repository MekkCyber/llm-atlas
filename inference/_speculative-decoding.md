# Speculative Decoding

*Taxonomy — "guess then verify" decoding, where a cheap draft proposes tokens that a target model accepts or rejects in parallel.*

**TL;DR:** Autoregressive decoding is memory-bound: each token needs one forward pass of the target model. Speculative decoding runs a cheap **draft** model (or draft head) that proposes several tokens; the target model then verifies all of them in a *single parallel pass*. Accepted tokens are kept; the first mismatch is corrected and the rest discarded. Speedups come from amortising the target model's weight-streaming cost across multiple accepted tokens.

**Related taxonomies:** [_llm-serving.md] *(not yet)*
**Depth files covered here:** [specfold](specfold.md)

---

## The problem

LLM decode is bottlenecked by memory bandwidth — every weight must be loaded once per generated token, regardless of how much compute each FLOP uses. A single forward pass over a 70B model to produce one token leaves tensor cores idle most of the time. Speculative decoding fills that idle budget by verifying many candidate tokens at once.

## The shared pattern

Every variant has the same shape:

1. **Draft.** Produce $k$ candidate tokens cheaply — a small draft model, a self-draft head (Medusa/EAGLE), a lookup table (prompt-lookup), or diffusion LM parallel sampling.
2. **Verify.** The target model's single forward pass computes probabilities for all $k+1$ positions in parallel.
3. **Accept.** Walk the drafted tokens left to right, accepting under a rejection-sampling rule that preserves the target's distribution. First rejection truncates the run; the target's own sample fills that position.

Speedup = (expected accepted tokens per cycle) / (per-cycle cost ratio). Winning draft models are *cheap* and *correlated with the target*.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Draft-model (vanilla) | Separate small LM proposes tokens | Needs a well-aligned small model | Target model is large; draft is cheap and highly correlated. |
| Medusa | Target gets several extra decoding heads | Extra parameters/training | Target supports fine-tuning; you want zero extra model. |
| EAGLE / EAGLE-2 | Draft is a lightweight autoregressive head on top of target's hidden states | Draft training; limited to one family | High correlation drafts with modest cost. |
| Prompt-lookup | Draft tokens come from the input prompt (n-gram match) | Only works when prompt likely repeats | Agent / code tasks with lots of repetition. |
| [specfold](specfold.md) | Fold redundant branches of diffusion-LM speculative verification via residual gating | Diffusion-LM only; requires custom Triton kernel | Multi-branch diffusion LMs. |

## How to choose

Default to **draft-model speculative decoding** when a well-aligned small model exists (same tokenizer, similar family) — simplest, most robust gains. Move to **EAGLE/Medusa** when training the target is on the table and you want zero extra models. Use **prompt-lookup** on narrow agent/code pipelines. For **diffusion LMs**, standard AR speculative decoding ports badly — the multi-branch verification tree needs structure-aware tricks like SpecFold.

Speculative decoding stacks with KV-cache tricks and continuous batching; it does not stack trivially with routing across different model families (cache ownership becomes painful).

## Adjacent but distinct

- **MoE expert routing** — routes inside one model, not between a draft and a target.
- **Chain-of-thought short-cutting** — reduces the number of tokens generated, not the cost per token.
- **KV-cache compression** — orthogonal; reduces per-token memory, not per-token weight-streaming.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2022.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., 2023.
- Paper: *EAGLE / EAGLE-2* — Li et al., 2024.
- Paper: *SpecFold* — 2026 — [arXiv:2610.04875](https://arxiv.org/abs/2610.04875).
