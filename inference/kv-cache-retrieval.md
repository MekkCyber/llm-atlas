# KV-Cache Retrieval
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In long-horizon autoregressive generation, the KV cache is bounded, so old context is evicted and drift sets in when earlier state becomes relevant again. **KV-cache retrieval** brings the relevant past chunk *back into* the KV cache — instead of re-prompting or reconstructing state via retrieval-augmented text. Training-free when correspondences (pose, time, semantic key) are available.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [../agents/_context-management.md](../agents/_context-management.md), [../agents/context-compaction.md](../agents/context-compaction.md)

---

## What it is

A KV-cache management strategy: past chunks (both the tokens and their computed key/value tensors) are *stored* and later *re-inserted* into the active KV cache when the generation revisits their subject. Distinct from RAG (which re-inserts *text tokens* to be re-encoded) and from prefix caching (which never evicts). The correspondence signal — how to know which chunk to bring back — is provided externally: a game-engine pose match, a timestamp, or a similarity key.

## How it works

- **Chunk store.** As the generator runs, evicted KV chunks (keys, values, and their tokens) are archived along with a correspondence key.
- **Retrieval trigger.** At generation time, some external signal — same 3D pose, same entity, same timestamp region — flags a stored chunk as relevant.
- **Re-insertion.** The retrieved chunk's K/V tensors are placed back into the KV cache. No re-encoding of tokens; no additional prefill pass. This is the cheap variant of "long-term memory."
- **Attention biasing.** Optionally, at the token level, attention is biased toward the retrieved chunk's spatially/semantically corresponding regions, further sharpening the connection to past state.

## Why it matters

Long-horizon autoregressive generation across many domains — video world models, agent trajectories, streaming assistants — hits the same wall: bounded KV → forgotten history → drift. Framing "loop closure" as *retrieval into the KV cache* is a generalizable pattern and, crucially, doesn't require training when the correspondence is available for free (game engines' pose+depth output, chat systems' turn boundaries, agent systems' entity IDs).

## Gotchas & tricks

- **Positional-encoding mismatch is the failure mode.** Re-inserted K/V tensors carry their original positions; if the model relies on relative or absolute positions in ways the retrieval breaks, quality collapses.
- **Correspondence quality is a first-class dependency.** A wrong retrieval is worse than no retrieval — you actively bias the model toward the wrong state.
- **Store size is a real cost.** Full K/V tensors are large; quantize or subsample if you're storing many chunks.
- **Not universally applicable.** Best when the domain naturally provides a correspondence signal (game engines, timestamped events); harder in open-ended text.

## Sources

- Paper: *Closing the Loop: Training-Free Revisit Consistency for Autoregressive Generative Rendering* — Ma, Liu, Huang, Jiang (Roblox / Penn State), 2026 — [arXiv:2607.21848](https://arxiv.org/abs/2607.21848).
