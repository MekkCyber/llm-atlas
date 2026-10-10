# Soft-Token Memory for Frozen LLM Agents
*Depth — residual soft tokens that supplement textual summaries so a frozen LLM approximates full-context behaviour.*

**TL;DR:** Agent context windows cap how much history fits in the prompt; textual summaries drop information the frozen base model would otherwise use. **Soft-token memory** trains a small neural network to produce a bounded sequence of continuous "memory tokens" that get appended after the summary. Conditioned on summary + history, these soft tokens push a *frozen* LLM's output distribution back toward the full-context one — a residual connection along the sequence dimension. Introduced by REMORY (2026).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [reflective-rulebooks.md](reflective-rulebooks.md) · [procedural-world-generation.md](procedural-world-generation.md)

---

## What it is

A compaction mechanism for long-horizon LLM agents that keeps the base model frozen. Where naive summarisation produces a shorter textual history, soft-token memory produces a textual summary *plus* a short bundle of continuous tokens that live in the embedding space, learned to carry the information the summary lost.

Think of it as **prefix tuning for memory**: a tiny, trainable module writes a per-episode prefix (actually a suffix after the summary) that steers the frozen base model.

## How it works

Setup: a long history $H$, a textual summary $S$ produced by any summariser, and a frozen base LLM $f$. The goal is $f(S \,\|\, M) \approx f(H)$ where $M$ is a short sequence of soft tokens.

A **memory network** $g_\phi$ (small, trainable) ingests $S$ (and optionally $H$ during training) and emits $M = g_\phi(S) \in \mathbb{R}^{k \times d}$ for $k \ll |H|$ positions in the base model's embedding dimension $d$.

Training objective: minimise distributional distance (KL or reverse-KL on next-token logits) between $f(H)$ and $f(S \,\|\, M)$ on held-out continuations. Only $\phi$ moves; the base model's weights never change.

At inference, drop $M$ right after the summary; downstream tool-use, reasoning, or retrieval all run on the standard base model, which simply sees extra "tokens" in its context.

## Why it matters

- **Frozen-base compatibility.** Works with any API model or weight-frozen deployment: no fine-tuning, no new serving stack.
- **Residual framing.** Treating memory as a residual on top of a summary (not a replacement) sidesteps the lossy-summary problem without blowing the token budget.
- **Significant compression ratio.** REMORY approaches full-context performance on SummHay using **5.2%** of the input positions, with agent benchmarks showing fewer repeated tool outputs and tool errors.

The pattern generalises: soft-token prefixes for KV compression, for persona conditioning, for scratchpad externalisation — any case where you want a trainable, bounded-length conditioning signal next to a frozen model.

## Gotchas & tricks

- Training requires paired (full-history, summary) data; synthetic truncations from long documents work well but limit transfer to real agent traces.
- Soft tokens are *brittle to tokeniser mismatch* — they live in the base model's embedding space and transfer poorly across model families.
- The memory network should be *small enough to deploy next to the API* but large enough to be expressive. REMORY uses a few-layer transformer.
- Watch for *memory drift*: when memory tokens are updated across steps, errors compound; schedule periodic re-summarisation.

## Sources

- Paper: *REMORY: Learning Residual Memory for Context Compaction* — Xia et al., 2026 — [arXiv:2610.11287](https://arxiv.org/abs/2610.11287).
