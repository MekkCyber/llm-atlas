# Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A cheap **draft model** guesses the next `k` tokens; the expensive **target model** verifies them in a single forward pass. Accepted tokens are kept, the first rejected token is re-sampled from the target, and generation continues. Same output distribution as target-only decoding — but wall-clock speedup up to `k×` when the draft is accurate. Standard inference optimization since 2023; used in vLLM, SGLang, and every frontier serving stack.

**Prereqs:** [kv-cache.md](kv-cache.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [sparse-kv-prefetch.md](sparse-kv-prefetch.md), [../pre-training/mtp.md](../pre-training/mtp.md)

---

## What it is

Autoregressive decoding is memory-bound: each step loads all model weights into HBM to produce one token. If a smaller model could *guess* the next few tokens correctly, the big model could **verify** those guesses in one forward pass — paying essentially the same memory-bandwidth cost as one step, but retiring `k` tokens.

Speculative decoding formalizes this: a draft distribution `q(x)` proposes candidate tokens, the target distribution `p(x)` scores them, and a rejection-sampling rule (accept with probability `min(1, p(x)/q(x))`) guarantees the emitted sequence is distributed identically to samples from `p`.

## How it works

At each generation cycle:

1. **Draft.** Run the small draft model autoregressively for `k` steps (typically `k = 4–8`) to produce candidate tokens `x̃_1, ..., x̃_k` and their draft probabilities `q_i(x̃_i)`.
2. **Verify.** Run the target model on the concatenation `[prefix, x̃_1, ..., x̃_k]` in a single forward pass. This yields target probabilities `p_i(·)` at every position — the whole point of the technique.
3. **Accept-reject.** For each `i` in order, accept `x̃_i` with probability `min(1, p_i(x̃_i)/q_i(x̃_i))`. Stop at the first rejection.
4. **Resample.** At the rejection point, sample one replacement token from the residual distribution `p_i − q_i` (clipped and renormalized). Emit all accepted tokens plus this replacement.
5. **Loop.** Continue from the new frontier.

Common draft variants: (a) a smaller, same-family model (Llama-7B drafts for Llama-70B); (b) an early-exit head on the target model; (c) tree-structured drafts (Medusa / EAGLE) that verify multiple candidate branches in a single forward.

## Why it matters

- **Wall-clock speedup at zero quality loss.** Because the rejection-sampling rule preserves the target distribution, outputs are indistinguishable from ordinary target-model decoding. Reported speedups: 1.5–3× on typical workloads, up to 5× on repetitive text.
- **Cost model.** The extra draft forward passes are cheap; the win is verifying many tokens per target forward. Effective speedup ≈ `avg_accepted / (1 + draft_cost/target_cost)`.
- **Standard building block.** vLLM, SGLang, TGI, and TensorRT-LLM all ship spec-decode support out of the box.
- **The lookahead is reusable.** Recent work (OasisKV) uses the draft's lookahead as a *prediction of which KV blocks will matter*, extending spec-decode's utility beyond token generation into memory management.

## Gotchas & tricks

- **Draft quality dominates.** A weak draft with 30% acceptance rate barely helps — the target still runs almost every step. A good draft (>70% acceptance) can 2–3× throughput.
- **Draft must share tokenizer with target.** Different tokenizers break the position alignment on which verification depends.
- **Batching interacts messily.** Different sequences in a batch accept different numbers of tokens per step — vLLM and friends handle this with careful padding or per-sequence step counts.
- **Tree-structured drafts (Medusa, EAGLE) improve acceptance** by trying many candidate branches, at the cost of a larger verification forward. Usually a net win at low-to-medium batch sizes.
- **Temperature > 0 changes the sampling rule.** The `p(x)/q(x)` acceptance rule assumes temperature-`T` targets; both distributions must be scaled consistently.
- **Self-speculation via MTP.** DeepSeek-V3 uses its Multi-Token Prediction head as the draft — same weights, one extra head, ~1.8× speedup with no separate draft model.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023 — the original algorithm and correctness proof.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., DeepMind, 2023 — parallel independent derivation.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024 — tree-structured self-drafting.
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024 — feature-level drafting.
- Paper: *DeepSeek-V3 Technical Report* — 2024 — MTP as a self-drafting mechanism.
