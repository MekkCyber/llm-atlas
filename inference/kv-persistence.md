# KV Persistence (Galahad-KV)

*Depth — materialise the per-block KV cache to local NVMe so any block can be restored byte-exact at serve time with zero recompute and constant GPU memory.*

**TL;DR:** Transformer serving systems recompute the KV state of a prompt every time the request is received. For very long contexts (millions of tokens) this is prohibitive and context-window limits apply. Galahad-KV writes the KV state for every ~16k-token block to encrypted local NVMe, and later restores any block **byte-exact**, with zero recompute. The attention math is unchanged; the KV is just paged in and out of storage. Measured on 50-million-token streams with one H100 and Gemma 4 12B / 31B.

**Prereqs:** [mla](../architectures/mla.md) *(adjacent KV-compression angle)*
**Related:** [_speculative-decoding](_speculative-decoding.md) *(adjacent inference optimisation)*

---

## What it is

Treat the KV cache like a page cache. The attention state for a block of ~16k tokens is a deterministic function of the model weights and the token sequence; compute it once, serialise, and restore from disk when the same prefix is re-encountered. The restored KV is bit-identical to recomputation (same model, same input → same floats), so the model's downstream behaviour is unchanged.

Three concrete pieces:

- **Block layout.** Context is split into contiguous ~16k-token blocks. Each block's KV goes to its own encrypted NVMe file.
- **Byte-exact restore.** Floating-point KV tensors are stored in their native dtype; no quantisation, no re-encoding. Restore is deterministic.
- **Cheat-resistant retrieval protocol.** For evaluation, planted facts are probed at random depths; the protocol prevents trivial "the answer is in the last block" cheats.

## How it works

Serving-side flow for a long-context request:

1. Request arrives with a reference to a (potentially 50M-token) corpus and a query.
2. The query is answered attending over whichever blocks are needed; blocks not yet in GPU memory are **restored from NVMe** on demand.
3. GPU memory holds only a working set of blocks at any time; the rest live on disk.
4. Any new tokens the model writes are KV-cached and persisted to new blocks.

Because restore is byte-exact, there is no accuracy delta between "KV from recompute" and "KV from NVMe". The win is latency and GPU energy.

## Why it matters

- Removes the context-window limit from the serving layer. 50M tokens is demonstrated; the paper's measurements extrapolate.
- Restore is **2.8× – 4.3× faster** than recompute of the same block, and uses **8.8× – 12.3× less GPU energy**.
- **GPU memory stays flat** across the entire stream; disk grows linearly.
- Gemma 4 12B recalls 82 / 100 planted facts at depths up to 50M tokens with zero hallucinations; 31B scores 98 / 100.
- Public reproduction on vLLM + one H100 using the `galahad-kv` package.

## Gotchas & tricks

- **NVMe throughput, not capacity, is the bottleneck.** Random restores from a 50M-token corpus hit NVMe at random offsets; align block size to NVMe page size.
- **Encryption-on-write isn't free.** The paper reports the overhead is dwarfed by the compute win, but on slower disks it matters.
- **Model-change invalidation.** The KV is tied to the exact model weights. A new checkpoint invalidates every stored block.
- **Attention sinks and long-context tricks.** Positional-encoding alignment must survive restore; the paper restricts to models whose attention is translation-invariant at block granularity.
- **Not retrieval.** The attention mathematics is unchanged — the model attends over *every* token in the restored blocks, not a filtered subset.

## Sources

- *Real Long-Term Memory for AI: A 50-Million-Token Window That Is Faster and Cheaper Than Recompute* — Corbenic AI — 2026 — [arXiv:2610.10845](https://arxiv.org/abs/2610.10845)
