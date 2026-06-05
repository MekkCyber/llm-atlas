# Streaming multi-agent reasoning (StreamMA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Multi-agent reasoning systems traditionally run "generate-then-transfer": agent N waits for agent N-1 to finish before starting. StreamMA pipelines them by streaming each reasoning step downstream as soon as it is produced. This reduces end-to-end latency from $O(\text{depth})$ toward $O(1)$ — and, surprisingly, *also improves* effectiveness, because early reasoning steps are more reliable than late ones and partial trust is better than full trust. From Yang et al., 2026.

**Prereqs:** [README.md](README.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)
**Related:** [../post-training/reasoning/length-penalty.md](../post-training/reasoning/length-penalty.md)

---

## What it is

A multi-agent reasoning pipeline is a chain (or tree, or graph) of LLM agents where each consumes the previous agent's output. Three protocols are possible:

- **Serial.** Agent $i$ runs to completion; pass output to agent $i+1$. End-to-end latency $\sum_i T_i$.
- **Single.** One agent does everything. Latency $\max_i T_i$, but no diversity.
- **Stream.** Agent $i+1$ starts consuming agent $i$'s tokens as they are emitted; both run concurrently. Latency $\approx \max_i T_i + \text{depth} \cdot \text{step}$.

StreamMA is the stream protocol made explicit, with a closed-form analysis of when it wins.

## How it works

1. **Token-level pipelining.** Each agent's output is a stream. Downstream agents subscribe and begin reasoning as soon as enough tokens have accumulated to be useful (a configurable prefix).
2. **Early-step bias.** Because long CoTs are non-uniform in quality — early steps are more reliable than late ones — operating on the streaming prefix is *better* than operating on the full completion, since late exploration noise is filtered out by truncation.
3. **Closed-form protocol analysis.** The paper derives:
   - effectiveness ordering (stream ≥ serial in expectation, under a "early steps more reliable" assumption),
   - speedup upper bound (≈ pipeline depth for compute-bound stages),
   - cost ratio (stream pays the same total tokens but spreads them across wall-clock).
4. **Step-level scaling law.** Orthogonal to scaling agent count, scaling *per-agent steps* improves both effectiveness and efficiency under streaming. New axis for multi-agent compute.

## Why it matters

- **Latency.** Agent pipelines have been quietly bottlenecked by serial waiting; streaming cuts wall-clock without quality cost.
- **Reliable-prefix exploitation.** A free quality win from truncation, validated empirically and explained theoretically. Counterpoint to "more thinking is always better."
- **New scaling dimension.** Step-level scaling is composable with agent-count scaling, suggesting more dimensions for compute-optimal multi-agent design than was previously appreciated.

Reported across 8 reasoning benchmarks (math/science/code), two frontier LLMs (Claude Opus 4.6, GPT-5.4), and three topologies (Chain / Tree / Graph): avg +7.3 pp over the best baseline; max +22.4 pp on HMMT 2026.

## Gotchas & tricks

- **Prefix-length tuning.** Subscribing too early (single-token granularity) means downstream agents waste effort on partial fragments; too late and you lose the pipelining benefit. Per-task tuning matters.
- **Topology matters.** Streaming gains are largest on chains; trees benefit when sibling agents can each consume the same parent stream; graphs need explicit barrier reasoning.
- **Doesn't help single-agent runs.** Stream is a *multi-agent* speedup; single-agent CoT has no pipelining axis.
- **Cancellation semantics.** When an upstream agent retracts (rare but possible in tree-of-thoughts), downstream consumers must be able to back out. Engineering, not theory.

## Sources

- Paper: *Streaming Communication in Multi-Agent Reasoning* — Yang et al., 2026 — [arXiv:2606.05158](https://arxiv.org/abs/2606.05158).
- Related: tree-of-thoughts, debate-style multi-agent reasoning.
