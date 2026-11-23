# Mixture-of-Block-Attention (MoBA)
*Depth — sparse long-context attention that routes each query to a learned subset of KV blocks.*

**TL;DR:** Split the key/value sequence into contiguous blocks, compute a cheap score between a query and each block, and attend only to the **top-K blocks**. Reduces long-context attention from $O(L^2)$ toward $O(L \cdot K \cdot B)$ with learned relevance instead of locality heuristics. The mechanism underlying Kimi's long-context stack; "LightningMoBA" (2026) is the kernel that makes it fast on Hopper.

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [_moe](_moe.md)

---

## What it is

A sparse attention pattern where the sparsity is **learned and query-dependent**, not fixed (unlike sliding-window or stride-based attention). Each query picks the KV blocks it wants to look at based on a lightweight scoring function; non-selected blocks contribute zero.

## How it works

Partition the KV sequence of length $L$ into $B$ contiguous blocks of size $L/B$.

1. **Score.** For each query $q$ and each KV block $j$, compute a scalar score $s_{q,j}$ (e.g. dot product with a block summary, or a small MLP over a block's mean key).
2. **Select.** Pick the top-$K$ blocks by score. The original MoBA uses a fixed $K$; **LightningMoBA** replaces this with **threshold routing** — keep every block whose score exceeds a learned threshold, so $K$ varies per query and is large only when needed.
3. **Attend.** Standard attention restricted to those blocks:

$$
\mathrm{Attn}(q) = \sum_{j \in \mathrm{Top\text{-}K}(q)} \mathrm{softmax}(q K_j^\top / \sqrt{d}) V_j.
$$

4. **Kernel-level optimizations (LightningMoBA).**
   - **KV-centric execution:** reorder the inner loop around (query head, KV block) so each KV block is streamed once and reused across heads instead of per-query loads.
   - **Runtime pipelining:** overlap block scoring, KV loads, and the attention compute using Hopper's async copy engines. Scoring hides behind loading; loading hides behind compute.

LightningMoBA reports mean RULER rising from 74.42 to 75.28 while matching Top-16 retention, and **1.20–1.60× end-to-end generation speedup** over MoBA at matched quality.

## Why it matters

- **Scaling sparse attention to practice.** MoBA is cheap in theory; without the kernel work it is often slower than dense flash-attention under 32k because the routing overhead dominates. LightningMoBA pushes the crossover point lower.
- **Query-dependent sparsity beats fixed patterns.** Sliding-window and stride-based attention fix sparsity before seeing the data. Needle-style recall tasks punish this. Learned routing fixes it, at the price of the scoring step.
- **Composes with hybrid attention.** A hybrid stack can use MoBA in its sparse layers and full attention in the few preserved full-attention layers.
- **Frontier open-source adoption.** The Kimi family builds on MoBA in production long-context runs; a faster kernel lowers the operating cost.

## Gotchas & tricks

- **Scoring function matters.** Mean-key summaries lose information; a small MLP or a learned projection can recover long-tail recall at a small overhead.
- **Threshold routing needs a calibration step.** The threshold must track the model's score distribution per layer; a single global threshold loses quality.
- **Block size is a knob.** Too small and the scoring overhead dominates; too large and recall drops. 512 tokens per block is a common operating point.
- **Attention head sharing.** KV-centric execution relies on heads sharing their routing choice over a block; if head-independent routing is required, the KV-reuse win shrinks.
- **Hopper-specific.** The async copy engine exploitation is architecture-specific. Ampere / MI300 kernels need different pipelines.

## Sources

- Paper: *MoBA: Mixture of Block Attention for Long-Context LLMs* — Moonshot AI (Kimi), 2025.
- Paper: *Accelerating Mixture of Block Attention via Dynamic Routing, KV-Centric Execution and Runtime Pipelining* (LightningMoBA) — 2026.
