# Infinite-horizon video rollout (Echo-Infinity)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Autoregressive video generation usually drifts or runs out of context within minutes. Echo-Infinity combines *learnable evolving memory queries* with a *Unified Relative RoPE* recipe so that a single AR video model can sustain real-time rollouts for 24 hours (>1.3M frames) without drift. The memory queries summarize the past in a fixed-size token bank; the unified RoPE gives consistent positional semantics across condition, memory, and rolled-out tokens. From Bian et al., 2026.

**Prereqs:** [../fundamentals/rope.md](../fundamentals/rope.md), [README.md](README.md)
**Related:** [../fundamentals/_positional-encoding.md](../fundamentals/_positional-encoding.md)

---

## What it is

AR video models predict the next frame (or frame chunk) conditioned on past frames and a text prompt. Two failure modes set in past the training context length:

- **Truncation drift.** Sliding-window context discards old frames, so identity, scene layout, and physics gradually decohere.
- **Quadratic blow-up.** Keeping the full history scales attention cost super-linearly; real-time generation becomes infeasible after a few minutes.

Echo-Infinity proposes a fixed-size *evolving memory* of learnable queries that summarizes the rolled-out past, plus a positional encoding scheme that keeps the relative-position semantics consistent regardless of whether a token is a condition, a memory slot, or a recent frame.

## How it works

1. **Memory queries.** A fixed-size set of learnable query tokens $M = (m_1, \ldots, m_K)$ accompanies the running rollout. After each generation chunk, an *update step* refreshes $M$: cross-attention reads from the latest frames, distilling them into the bank.
2. **Memory tokens participate in attention.** At every generation step, attention attends to text condition + $M$ + recent window. Cost is bounded by $|M| + \text{window}$, regardless of total rollout length.
3. **Unified Relative RoPE.** Positions of condition tokens, memory tokens, and rollout tokens are mapped into a single relative-coordinate space, so attention scores don't depend on which absolute time the rollout has reached. This is what enables drift-free continuation past the training context.
4. **Streaming inference loop.** Each new chunk is generated, memory is updated, oldest non-memory tokens are dropped. The model never sees more than the memory + window at once.

## Why it matters

- **Truly infinite rollouts.** 24 hours and >1.3M frames sustained — well beyond what prior AR video models achieve (typically minutes).
- **Real-time at constant cost.** Per-step compute is constant in elapsed rollout time, making AR video viable for game engines, AR companions, and surveillance simulation.
- **Architectural template.** Evolving memory + relative positional unification is a pattern likely to repeat in long-horizon audio, agent rollouts, and any modality with strict latency budgets.

## Gotchas & tricks

- **Memory capacity sets the floor on quality.** Too few queries → memory smears, scenes blur over time. Too many → cost rises.
- **Memory-update schedule.** Updating every chunk is the default but optimal frequency is task-dependent.
- **Unified RoPE design choices matter.** Naive concatenation of three sub-spaces (condition, memory, frames) into one RoPE axis can leak positional cues between zones; the paper's recipe is non-trivial.
- **Training cost.** Long-rollout training is expensive even with memory; the model is trained on truncated rollouts and extrapolates via the relative positional design.

## Sources

- Paper: *Echo-Infinity: Learning Evolving Memory for Real-Time Infinite Video Generation* — Bian et al., 2026 — [arXiv:2606.04527](https://arxiv.org/abs/2606.04527).
- Related: RoPE (Su et al., 2021); world-model literature for long-horizon AR video.
