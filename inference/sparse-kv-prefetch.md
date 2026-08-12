# Lookahead-Guided Sparse KV Prefetch (OasisKV)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Keep only the *predicted-important* KV entries in HBM and stage the rest from host / remote memory tiers. Use the lookahead tokens from **speculative decoding** as a nearly-free predictor of which KV blocks the target model will touch next, prefetching them before decode arrives. On vLLM: 1.69× throughput at 0.1 pt accuracy loss on reasoning workloads; up to 2.1× on multi-GPU long-context serving.

**Prereqs:** [kv-cache.md](kv-cache.md), [speculative-decoding.md](speculative-decoding.md)
**Related:** [../architectures/mla.md](../architectures/mla.md), [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

Two observations glued together. **(1) Decode-time attention is sparse:** at each generation step, only a small fraction of past KV entries carry non-negligible attention weight. **(2) Speculative-decoding drafts already compute a lookahead:** the draft model runs `k` steps ahead of the target, so its queries are a preview of what the target will attend to next.

OasisKV exploits both. HBM stores only a working set of KV blocks; the rest live in host or remote memory. A background attention pipeline uses draft queries to predict which KV blocks the target will need `k` steps from now, and prefetches them before the target's decode step actually runs.

## How it works

- **Storage tiering.** Full per-token KV cache lives in host or remote memory (cheap, abundant). HBM keeps a working set sized by a fixed budget (e.g. 2048 tokens per sequence).
- **Importance predictor.** During spec-decode's draft phase, the paper runs an efficient sparse-attention pass with the draft's queries against the *compressed* KV metadata in HBM to identify the top-k relevant blocks per layer.
- **Async prefetch.** Predicted-important blocks are pulled from host/remote memory into HBM through a background pipeline that overlaps with the target's own compute.
- **Verification.** When the target model runs its verification forward, the working set contains the KV entries the sparse prediction said mattered. If the prediction was wrong, standard fallback logic recovers correctness (at the cost of one stall).

## Why it matters

- **Breaks the HBM ceiling for long-context serving.** With 2048 tokens of HBM-resident KV, the paper serves sequences of ≥32K tokens with 0.7-point accuracy loss vs full attention. The rest lives in cheap memory.
- **Uses spec-decode's existing lookahead — no new predictor to train.** Prior sparse-KV work either lost accuracy (heuristic top-k) or trained a separate learned predictor. OasisKV gets the "what will matter" signal for free.
- **Composes with the rest of the vLLM stack.** Paged attention, continuous batching, prefill-decode disaggregation — all still work. This is a memory-tier addition, not a replacement.
- **Concrete numbers.** 1.69× throughput on reasoning workloads at 0.1 pt accuracy loss. Up to 2.1× on multi-GPU long-context serving. Under prefill-decode disaggregation: ~2× throughput with 6.5–9.7× less KV admitted per request and 2.2–2.6× less decode-node host memory than full KV transfer.

## Gotchas & tricks

- **Prefetch bandwidth must exceed decode consumption.** If the host↔HBM link is the bottleneck, the sparse-serving win evaporates. The paper's numbers assume modern NVLink / PCIe generations.
- **Working-set budget is a tunable.** Too small → accuracy loss on hard queries; too large → most of the win is gone. 2048 is the paper's sweet spot on reasoning workloads.
- **Wrong-prediction stalls are recoverable but not free.** A mispredicted block forces a synchronous fetch; a workload with erratic attention patterns will pay this cost often.
- **Depends on spec-decode being enabled.** No spec-decode, no lookahead, no free predictor. Standalone OasisKV would need a different importance signal.
- **Not orthogonal to MLA.** MLA already compresses KV; OasisKV chooses which compressed blocks to keep resident. Both wins compose, but a MLA model has less to gain per prefetched block.

## Sources

- Paper: *OasisKV: Scaling In-Decode KV Cache Beyond HBM with Lookahead Sparse Prefetching* — Microsoft Research / Imperial, 2026.
- Paper: *Efficient Memory Management for Large Language Model Serving with PagedAttention* — Kwon et al., 2023 — the vLLM base OasisKV builds on.
- Related: OasisKV is a lookahead consumer; see [speculative-decoding.md](speculative-decoding.md) for the lookahead source.
