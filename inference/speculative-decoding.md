# Speculative Decoding
*Depth — a family of inference-time tricks that amortize LLM decode latency by letting a cheaper proposer predict multiple tokens that a verifier accepts in parallel.*

**TL;DR:** Autoregressive decoding is memory-bandwidth bound — one forward pass per token. **Speculative decoding** adds a *proposer* that emits `k` candidate next tokens, and a *verifier* that accepts a prefix of them in one forward pass. Classical speculative decoding (Leviathan 2023) uses a small draft model as proposer and the target model as verifier. Modern variants fold the proposer into the target model (Medusa heads, EAGLE, lookahead decoding) or reverse the roles (ProVe, 2026 — large proposer, small verifier). The common ceiling is bounded by the target model's token-level acceptance rate; aggressive variants trade accuracy for throughput.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md) · [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md)

---

## What it is

In a standard autoregressive decode step, the target model `T` samples one token given context `c`, pays one full forward pass, and advances the state. At batch size one, the GPU's matmul units are underutilized — the forward pass is bound by loading the weights, not by the matmul.

Speculative decoding exploits that slack: produce `k` **candidate tokens** cheaply, then run one forward pass of `T` to score all `k` positions in parallel. Any candidate whose posterior matches `T`'s distribution is accepted; the first mismatch terminates the prefix. In expectation, each verifier pass advances the state by `α · k + 1` tokens where `α ∈ [0, 1]` is the mean per-position acceptance rate — potentially several tokens per expensive forward pass.

**Two design axes:**
- **Who is the proposer?** A separately trained draft model; auxiliary heads on `T` itself; a fine-tuned version of `T`; `T` itself used with a look-back window.
- **Who is the verifier?** `T` itself (lossless speculative decoding); a smaller model (lossy — trades quality for speed).

## How it works

### Classical draft-and-verify (Leviathan, 2023)

1. Draft model `D` (small, same vocabulary as `T`) proposes tokens `x_{t+1}, x_{t+2}, …, x_{t+k}` autoregressively. `D`'s cost per token is small.
2. `T` runs *one* forward pass over the prompt + `k` draft tokens, producing posterior distributions `p_T(· | prefix)` at each position.
3. Accept each draft token `x_{t+i}` with probability `min(1, p_T(x_{t+i}) / p_D(x_{t+i}))`. Stop at the first rejection.
4. At the rejection point, sample from the corrected distribution `max(0, p_T − p_D)` normalized. Resume drafting.

This procedure is **lossless**: the output distribution equals `T`'s under the sampling scheme. Throughput wins come from accepting `α · k + 1` tokens per `T` forward pass.

### Self-draft variants

- **Medusa** adds `k` extra prediction heads to `T`, each predicting tokens `t+2, t+3, …, t+k+1` directly. The draft is a parallel forward pass of these heads; verification is one standard forward pass. No separate draft model.
- **EAGLE** uses a lightweight auto-regressive head on top of `T`'s features for drafting; empirically higher acceptance rates than Medusa at similar cost.
- **Lookahead decoding** fills the draft from recent n-grams in the context (Jacobi/lookahead style) — no trained draft, but lower `α` on open-ended text.

### Role reversal (ProVe, 2026)

Fine-tune the **large** target to propose entire token blocks in one forward pass; use a **smaller** model as verifier. Lossless with a 32B proposer gives 2.5× throughput; a 3B verifier at ≤2.6 pt accuracy drop gives 2.7×. Removes the need for a separately trained draft model.

## Why it matters

- **2–3× throughput at single-user inference** for most mainstream models. The gap to AR shrinks at large batch size (where compute becomes the bottleneck, not memory), so the win is biggest in latency-bound serving.
- **No model-quality cost** in the lossless variant — the output distribution is provably `T`'s.
- **Compounds with other tricks.** Works alongside continuous batching, KV-cache paging, quantization — it's a decoder-loop trick, orthogonal to the model internals.

## Gotchas & tricks

- **Acceptance rate dominates.** If `α` is low (hard domain, mis-trained draft), overhead of the draft wipes out the gain. Evaluate `α` before committing.
- **Vocab mismatch kills it.** Draft and target must share a tokenizer. Different tokenizers means draft tokens cannot be verified directly.
- **Memory cost.** Verifier processes `k` tokens' worth of KV cache in one pass; batch sizes that fit at `k=1` may not fit at `k=8`.
- **Sampling temperature.** High temperature lowers `α` because `T`'s distribution is flatter; sometimes worth lowering `k` when sampling hot.
- **Lossy variants need calibration.** Smaller-verifier speculative decoding (ProVe's 3B verifier) trades a small quality drop for speed — measure the drop on your actual eval before shipping.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., Google, 2023 — the canonical draft-and-verify algorithm.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024.
- Paper: *ProVe: Proposer-Verifier Decoding* — Jin et al., 2026 — [arXiv:2610.27815](https://arxiv.org/abs/2610.27815).
