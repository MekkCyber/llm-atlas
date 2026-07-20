# KV-Cache Grafting
*Depth — deposit a byte-exact KV state once, restore it into fresh inference contexts.*

**TL;DR:** Compute the model's key-value cache once for a chunk of verified reference material, serialize it **byte-exactly**, and later re-inject it as the prefix state of a new query. Under a pinned deterministic configuration, the grafted logit vector is SHA-256-equal to a fresh recompute, KL divergence is zero, argmax agreement 100%. Weights don't change; accelerator budget doesn't change. Reframes prompt/prefix caching as a *bit-exact knowledge artifact* rather than a numerical approximation.

**Prereqs:** [../architectures/mla.md](../architectures/mla.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [../fundamentals/dca.md](../fundamentals/dca.md)

---

## What it is

Prompt caching, prefix caching, and RAG all rely on the observation that if you've already processed a long prefix once, you shouldn't reprocess it. Standard implementations reuse the KV cache but drift numerically across kernels, dtypes, or scheduler decisions — so the reused cache is *approximately* equivalent, not bit-exact.

KV-cache grafting takes the strongest form of this idea: serialize the KV state as a **byte-exact artifact**, distribute or store it, and re-load it into a fresh inference context. The paper reports SHA-256 equality of the resulting logits against a fresh recompute, zero KL, and 100% argmax agreement across 50 samples — under a pinned deterministic configuration (fixed layer order, deterministic kernels, controlled dtype).

## How it works

1. **Deposit.** Run the reference material through the model once; capture the KV tensors of every layer. Serialize them alongside model identity, tokenizer state, and the numerics configuration (kernel choice, dtype, RNG seed if any).
2. **Store.** The artifact is a byte-exact snapshot — content-addressable (e.g. SHA-256), portable, and safely reusable across sessions.
3. **Graft.** At query time, load the artifact into the fresh context's KV slots, then start decoding from the first uncached position. Positional encodings must be applied consistently with the deposit-time context.
4. **Verify.** In the paper's experimental setup, compare grafted logits against a fresh recompute: SHA-256 hash equality, zero KL, 100% argmax agreement.

Nothing in the weights changes. The model *thinks* it computed the prefix; the artifact makes that fiction bit-exact.

## Why it matters

- **Separates knowledge from weights.** A frozen 12B model can be extended with verifiable reference material without any fine-tuning or LoRA — the deposited artifact carries the knowledge.
- **Latency and cost win.** Skips the prefill for every query that reuses the artifact — the paper describes the frozen model as "measurably more capable and dramatically cheaper at the same time."
- **Bit-exactness is auditable.** Approximation-based caching has correctness gaps you can only spot with careful testing; byte-exact grafting can be verified with a hash.
- **Compositional.** Multiple artifacts (e.g. per-doc knowledge chunks) can in principle be concatenated in the KV dimension — subject to the positional-encoding caveat below.

## Gotchas & tricks

- **Requires pinned determinism.** The bit-exact claim depends on holding the numerics constant — kernel version, dtype, thread count, and layer order all matter. Off-recipe (different GPU family, different attention kernel) can break equality.
- **Positional encodings must be respected.** If the grafted KV state was computed at positions $[0, L)$, it must be reused at those positions; RoPE-based models require careful bookkeeping to compose multiple artifacts.
- **Storage cost is real.** A KV cache for tens of thousands of tokens is many gigabytes; artifact stores need dedup and compression to be practical.
- **Cross-model incompatibility.** Artifacts are tied to a specific model checkpoint; new fine-tunes invalidate them.
- **Not a general RAG replacement.** Grafting only helps for *pre-known* reference material; on-the-fly retrieval still needs a real search + prefill loop.

## Sources

- Paper: *Byte-Exact KV-State Grafting Turns a Frozen Small Model into a Verified-Knowledge Flywheel* — Sietse Schelpe (independent), 2026.
