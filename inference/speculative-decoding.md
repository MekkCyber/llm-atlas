# Speculative decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use a cheap **drafter** to propose several candidate next tokens in parallel, then have the main model **verify** them in a single forward pass. Accepted drafts return multiple tokens per step; rejected ones fall back to the main model's prediction. Lossless (exactly the same output distribution as the main model) and near-universally applicable to autoregressive decoding.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** none

---

## What it is

Autoregressive decoding emits one token per forward pass. The forward pass is dominated by weight loading from HBM — so the per-token cost is roughly constant whether you want 1 or 8 tokens of output. Speculative decoding exploits this: if a cheaper model's predictions often match the main model's, you can verify a short run of them *in one* main-model forward pass and keep them.

## How it works

Let `p` be the main (target) model and `q` a drafter (a smaller model, an earlier layer, or a learned head).

**Draft.** `q` generates `k` candidate next tokens `t_1 … t_k` autoregressively.

**Verify.** `p` runs a single forward pass on `[context, t_1, …, t_k]`, producing probabilities for each position.

**Accept / reject** per token using a Metropolis-style rule:
- Accept `t_j` with probability `min(1, p(t_j|context) / q(t_j|context))`.
- On rejection at position `j`, sample a replacement from the residual distribution `(p − q)_+`.

The arithmetic guarantees that the full process samples from `p` exactly. Expected accepted tokens per step is `≤ k` and depends on how well `q` approximates `p`.

## Why it matters

A 2–3× end-to-end decode speedup is routinely achievable on standard LLM inference workloads. Variants extend the idea substantially:

- **EAGLE** uses a learned head on top of the main model's last hidden state as the drafter.
- **Medusa** adds multiple drafter heads for parallel multi-position proposals.
- **Self-speculation** reuses an early exit from the main model as the drafter (no second model).
- **Speculative listening** ([speculative-listening](speculative-listening.md)) applies the pattern to full-duplex speech LMs where chunk idle time is the budget.

## Gotchas & tricks

- **Draft length `k` is a hyperparameter.** Too short → little gain; too long → low acceptance rate wastes verification compute. Typical `k ∈ [4, 8]` for text, `k ∈ [1, 4]` for code where acceptance is lower.
- **Drafter choice.** The drafter must be cheap *and* a decent approximation of `p`. A random draw is zero cost but zero acceptance; the main model is perfect but defeats the purpose.
- **Tokenizer parity.** Drafter and target must share a tokenizer, otherwise acceptance reduces to a lookup problem.
- **Not a throughput win.** Speculative decoding helps latency and latency-bound throughput. On a server already batched at saturated FLOPs, it does nothing.

## Sources

- Paper: Fast Inference from Transformers via Speculative Decoding — Leviathan et al., 2023.
- Paper: Accelerating LLM Inference via Speculative Decoding — Chen et al., 2023.
- Paper: EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty — Li et al., 2024.
