# Residual Memory for Context Compaction (REMORY)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Compact a long interaction history into a short natural-language summary plus a small set of *learned soft memory tokens* that act as the "residual" between the summary and the model's true hidden states. A frozen LLM then conditions on summary + memory tokens and reproduces full-history behavior at a fraction of the context cost (~5.2% of positions in the paper's benchmarks).

**Prereqs:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [token-level-routing.md](token-level-routing.md) · [../agents/README.md](../agents/README.md)

---

## What it is

Agents accumulate long interaction histories (tool calls, observations, prior turns). Natural-language summarization is readable but lossy; keeping the full history is behaviorally safe but expensive. REMORY argues these are not a two-way choice: keep the summary for *humans* and attach learned continuous tokens that carry the information the summary dropped — a lossy-plus-residual pair.

The LLM backbone stays frozen; only the memory-token embeddings (and a tiny producer network that generates them from the full history) are trained.

## How it works

- **Summarizer.** A text summarizer compresses the raw history into a short natural-language account.
- **Memory producer.** A small trainable module (treated as a soft-prompt generator) takes the full history and outputs $m$ continuous token embeddings.
- **Training signal.** The frozen LLM is fine-tuned on the task loss when it is given summary + memory, and the memory tokens are optimized to make the frozen model's behavior match its behavior on full history.
- **Deployment.** At inference, run summarizer + memory producer once per compaction event; the LLM then operates on summary + memory tokens.

## Why it matters

- Keeps a human-readable surface (the text summary) that lets operators inspect, edit, or redact.
- Preserves agent behavior where pure summarization causes drift (repeated tool calls, forgotten constraints).
- Compacts to ~5.2% of the original context positions on SummHay with better source attribution and comparable insight coverage, and reduces repeated tool outputs and tool errors on long-horizon agent benchmarks for two backbones.

## Gotchas & tricks

- Memory tokens are *model-specific*: a producer trained for one backbone does not transfer directly to another.
- The method expects the summary to be reasonably good — pathological summaries can push too much onto the memory tokens and overflow their capacity.
- Compaction is a discrete decision; policies for *when* to compact (every $k$ turns? on context pressure?) are not fixed by the method and should be tuned per deployment.

## Sources

- Paper: *REMORY: Learning Residual Memory for Context Compaction* — Xia, Chen, Ge et al., 2026 — [arXiv:2610.11287](https://arxiv.org/abs/2610.11287).
