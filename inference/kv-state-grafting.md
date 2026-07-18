# Byte-Exact KV-State Grafting
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Cache the per-layer KV state produced when a model processes "verified knowledge" once, then *graft* that state back into fresh inference contexts on demand. Under a pinned deterministic configuration, the grafted context produces logits **byte-identical** to a fresh recomputation over the original text — SHA-256 equality, zero KL, 100% argmax agreement. The model weights are unchanged. The trick amortizes the cost of expensive one-shot computations over long documents into reusable artifacts.

**Prereqs:** *(none — foundation KV-cache page not yet written; see [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md) for the underlying attention structure the cache stores state for)*
**Related:** *(none in KG yet; this is the first inference depth file)*

---

## What it is

A standard KV cache is *ephemeral*: it lives inside a running inference session, is written during prefill, read during decode, and discarded when the session ends. Byte-Exact KV-State Grafting reframes the cache as a **portable artifact**:

1. **Deposit.** Run the model over some source text once (a long document, a validated knowledge base, an evaluated tool trace). Serialize the resulting per-layer KV cache.
2. **Store.** Persist the artifact alongside the source (or on its own, if the source is confidential).
3. **Restore.** When a new inference session would benefit from the same context, load the artifact and *graft* it into the new session's per-layer KV cache — no reprocessing of the source text.
4. **Continue.** Decode from the grafted state as if the source had been part of the prompt.

The strong claim is byte-exact restoration: under a pinned deterministic configuration, the grafted state produces logits identical to freshly recomputing the source text — bit-for-bit, not just numerically close.

## How it works

The engineering problem is that modern accelerator kernels are non-deterministic in ways that quietly corrupt byte-exactness:

- **Reduction order in attention.** FlashAttention and its variants aggregate over sub-blocks; the aggregation order depends on block-scheduling decisions the runtime is free to change.
- **Non-associative float math.** Different reduction orders produce different bit patterns even when arithmetically equivalent.
- **Kernel selection.** Runtimes pick between kernels based on shape, hardware, and version — different kernels produce different bit patterns.

Byte-exactness requires pinning all of these:

- Pin the kernel (deterministic FlashAttention or exact reference kernel).
- Pin reduction order (single-block-per-token or explicit reduction-tree specification).
- Pin the accelerator's compute unit and driver version.
- Serialize the KV cache in the exact numeric format the kernel expects to read (no dtype casts on load).

Under those constraints, the deposit → restore round-trip is a byte-exact identity. Validation in the source paper: SHA-256 of the logit vector matches fresh computation across trials, KL divergence is zero across 50 samples, and argmax agreement is 100%.

## Why it matters

- **Amortizes long-context prefill.** A single 200k-token document that would take significant prefill time can be prefilled *once* and reused across arbitrary sessions.
- **Verified knowledge as a first-class artifact.** Compliance-critical scenarios (medical guidelines, legal reference material) benefit from cryptographically-checked reuse instead of relying on each session's runtime to produce the "same" behavior.
- **Deployment leverage without weight changes.** No retraining, no fine-tuning, no accelerator budget increase — the flywheel is entirely in software.
- **Sets a floor for reproducibility.** If byte-exactness is achievable under a pinned config, "close enough" numerical drift in deployment is a *choice*, not a hardware inevitability.

## Gotchas & tricks

- **Byte-exactness is fragile.** Any kernel upgrade, driver upgrade, or dtype change silently breaks it. Publish a pinned config alongside every artifact and treat drift as a security event.
- **KV artifacts encode weights indirectly.** Two artifacts produced by two different fine-tunes of the same base look nearly identical numerically but produce different logits — the artifact is *specific* to the model version. Version-tag artifacts by model checkpoint.
- **The engine is proprietary in the source paper.** The single-author paper describes the mechanism and validation but the engine and benchmark suite are gated. Community-side reproduction is the natural next step, and standard open kernels (FlashAttention-3 in deterministic mode) should be able to hit the same guarantee.
- **KV size ≠ token count.** The artifact size scales with `num_layers × num_kv_heads × head_dim × sequence_length`. For a 12B model with 40 layers and a 200k-token document, artifacts are large — cheaper than reprocessing, but not free storage.
- **Grafting position matters.** The grafted state must be inserted at the same position it was recorded at (or the position encodings must be re-run on the graft). Getting this wrong produces a working-looking model that hallucinates fluently.
- **Not RAG.** The artifact carries hidden-state activations, not retrieved text. There's no retrieval step, no chunking, no re-embedding — the artifact *is* the model's internal representation of the source.

## Sources

- Paper: *Byte-Exact KV-State Grafting Turns a Frozen Small Model into a Verified-Knowledge Flywheel* — Sietse Schelpe, 2026 — introduces the grafting mechanism and reports byte-exact round-trip validation.
