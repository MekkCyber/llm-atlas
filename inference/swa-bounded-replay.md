# SWA Bounded Replay
*Depth — a bound on sliding-window attention replay to shrink the persistent (SSD/host) KV cache.*

**TL;DR:** Sliding-window attention (SWA) recomputes attention edges when the window slides, which under naive persistence policies inflates the persistent KV cache stored on SSD or host memory. **SWA Bounded Replay** caps the replay per window, so DeepSeek-V4.1-Flash (2026) keeps its persistent KV footprint to roughly **1/8 of V4-Flash's** — the counterpart to the ~1/4 global (HBM) KV compression delivered by CSA2 + FP4.

**Prereqs:** [../architectures/mla.md](../architectures/mla.md), [../architectures/csa2.md](../architectures/csa2.md)
**Related:** [fp4-kv-cache](fp4-kv-cache.md), [../case-studies/deepseek-v4-1-flash.md](../case-studies/deepseek-v4-1-flash.md)

---

## What it is

An SSD/host-side deployment optimization. In agent workloads, part of the KV cache is persisted outside HBM: shared context across many turns, or context that spans multiple sessions. Sliding-window attention (which restricts attention to a fixed local window) simplifies HBM residency — only the current window needs to stay in HBM. But sliding-window means the *boundary* between windows must be replayed: recompute or reload the KV that overlaps two windows.

Naive replay policies (always keep the full window overlap in persistent cache) blow up persistent-cache size for long conversations. SWA Bounded Replay imposes a **cap on how much of each window's boundary is replayed**, trading a small amount of recomputation for a much smaller persistent cache.

DeepSeek-V4.1-Flash cites a ~8× reduction in persistent KV footprint vs. V4-Flash from this technique.

---

## How it works

Standard SWA setup: attention is restricted to the last $W$ tokens. When a new window advances by $\Delta$ tokens, the "replayed" portion is the overlap $W - \Delta$ tokens.

**Bounded replay** introduces a maximum $R \le W$ such that at most $R$ tokens of the overlap are re-loaded from persistent storage; the rest are either dropped (accepted as a small quality hit) or recomputed from a compressed sketch.

```
window boundary
  │
  ▼
[ ... older kv ... | overlap (W-Δ) | current window Δ ]
                        │
                        ▼
             clip to R:  [ replayed R | dropped/recomputed ]
```

The persistent SSD/host cache stores only $R$ tokens' worth of KV per boundary event, bounding the total persistent footprint regardless of conversation length.

The specific values of $W$, $\Delta$, and $R$ in V4.1-Flash aren't in the abstract; the effect (~8× reduction vs. V4-Flash) is.

---

## Why it matters

- **Persistent cache is the long-conversation cost driver.** For agents with long-running memory, persistent KV grows unboundedly under naive policies. Bounded replay caps it.
- **Complements HBM compression.** CSA2 + FP4 shrink the HBM-resident cache; SWA Bounded Replay shrinks the SSD/host-resident cache. Both matter for total serving cost.
- **Trades compute for storage.** Some replay boundaries must recompute what was dropped; on modern GPUs this is often the right tradeoff vs. SSD I/O bandwidth.

---

## Gotchas & tricks

- **Choose $R$ conservatively for reasoning agents.** Recomputation is expensive if it triggers frequently; drop-then-recompute at every window boundary hurts throughput.
- **Compressed sketches are worth trying.** A small SVD/pooling summary of dropped KV is cheaper to reload than raw KV, and better than dropping outright.
- **Interacts with FP4 KV.** Persistent-cache values can be at FP4 (extra compression) or at higher precision (better reload accuracy). Choose per-layer.
- **SSD I/O is the wall-clock ceiling.** Even with bounded replay, if the persistent cache still exceeds SSD read bandwidth, the technique doesn't help. Combine with async pipelining.
- **Boundary policy must be deterministic.** Non-deterministic replay policies break request-level reproducibility, which matters for agent debugging.

---

## Sources

- Paper: *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression* — DeepSeek-AI, 2026 — [arXiv:2609.19969](https://arxiv.org/abs/2609.19969).
- Case study: [deepseek-v4-1-flash](../case-studies/deepseek-v4-1-flash.md).
