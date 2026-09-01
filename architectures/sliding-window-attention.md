# Sliding Window Attention (with sinks)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Cap each token's attention window to the last $W$ keys (typical $W$ = 4K–8K), keep a small set of always-attended "sink" tokens at the start, and you get sub-quadratic decode with a bounded KV cache — **no training** required. As a *retrofit*, SWA+sinks matches or beats post-trained **linear-attention** retrofits on standard tasks and dominates them (2–10×) on long-context retrieval like Needle-in-a-Haystack and BABILong.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [mla.md](mla.md) · [../fundamentals/rope.md](../fundamentals/rope.md)

---

## What it is

Full attention has quadratic cost in sequence length and an unboundedly growing KV cache. SWA restricts each query to attend only to the last $W$ keys — cost per new token becomes $O(W)$ and KV memory stays bounded. **Attention sinks** are a small number of anchor tokens (typically the first 4) whose K/V are always in cache regardless of the window; they stabilize the softmax by giving it somewhere to dump probability mass when nothing else is relevant.

## How it works

- **Cache policy.** Keep the sink tokens permanently. For everything else, evict K/V older than $W$ steps back from the current token.
- **Attention mask.** At step $t$, query attends to tokens $\{0, \ldots, S-1\}$ (sinks) $\cup$ $\{t-W+1, \ldots, t\}$ (window). Everything else is masked out.
- **Retrofit vs from-scratch.** As a retrofit on a full-attention LLM, no training is needed — the model has already learned to attend, and truncating the window at inference time is a pure caching change. From-scratch SWA models train with the mask in place.
- **Interaction with RoPE.** SWA + RoPE composes cleanly if positions are indexed by the *absolute* token position, not the window offset — otherwise the model sees rotated K/Q at unfamiliar relative distances.

## Why it matters

- **No post-training** required for the retrofit case. Ships as a cache-and-mask change.
- On **long-context retrieval** (NIAH, BABILong), matches or beats post-trained linear-attention retrofits by **2–10×** — because linear attention compresses history irreversibly while SWA still exposes the recent context exactly.
- Extremely fast and low-memory: attention cost per token is constant in $W$, not in sequence length.

## Gotchas & tricks

- **Sinks are not optional.** Without them, when nothing in the window is relevant the softmax has nowhere to route mass and attention collapses onto the most recent token. Small ($S$ = 4) is enough.
- **Not a replacement for long-context recall beyond $W$.** Anything the model needs to "look up" that falls outside the window is lost. If you need retrieval past $W$, pair with an external retriever (RAG) or a memory-router variant.
- **Linear-attention comparison caveat.** SWA wins the *retrofit* comparison; linear-attention models trained from scratch (or with heavy post-training) can still be competitive. The paper's claim is specifically about post-training linear-attention retrofits.
- Typical values: $W$ = 4096 or 8192, $S$ = 4 sinks. Larger $W$ helps until it stops helping (task-dependent).

## Sources

- Paper: *Sliding-window beats linear attention* — Jolicoeur-Martineau et al., Samsung SAIT / Microsoft Research, 2026 — [arxiv](https://arxiv.org/abs/2608.28444)
- Prior: *Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2023 — the sinks recipe SWA-with-sinks builds on.
