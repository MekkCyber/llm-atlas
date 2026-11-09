# Latent tool-output compression
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Agent context bloats because tool outputs — search results, web pages, large JSON — are always serialized as text. Latent tool-output compression maps each raw observation to a handful of *continuous latent tokens* that occupy the agent's input embedding slots directly, skipping the detokenization-retokenization round trip. Trained with an Action-Preserving objective (KL between next-action distributions under full vs compressed context). Introduced in Coca-Cola (2611.06500).

**Prereqs:** [agent-memory](agent-memory.md)
**Related:** [harness-ensemble-training](harness-ensemble-training.md)

---

## What it is

A **drop-in replacement** for the standard "text tool output → tokenize → attend" pipeline. Tool outputs become *embeddings*, not strings; the human-readable version can still be logged but is not what the agent actually sees.

Three components:
- **Encoder** $E_\phi$: maps a raw tool observation $o$ to $K$ latent tokens $z_1, \ldots, z_K \in \mathbb{R}^d$ (same dimension as the agent's embedding space).
- **Slot insertion:** the latent tokens occupy context positions where the tool output would have been, so positional encodings and attention proceed as usual.
- **Action-Preserving Training:** the training objective (see below).

## How it works

Given a base agent LLM $\pi_\theta$ (frozen), raw tool output $o$, and compressed latent tokens $z = E_\phi(o)$:

1. Insert $z$ into the agent's context at the tool-output slot.
2. The agent produces a next-action distribution $\pi_\theta(a | \text{ctx with } z)$.

**Action-Preserving Training** aligns this to the ground-truth agent-with-raw-text:

$$\mathcal{L}_{\text{APT}} = \mathrm{KL}\big(\pi_\theta(a | \text{ctx with text } o) \;\|\; \pi_\theta(a | \text{ctx with } z)\big)$$

Only $E_\phi$ is trained; the agent LLM is unchanged. No reconstruction loss — the latent doesn't have to decode back to text, it just has to induce the same policy behavior.

$K$ (latent-token budget) is typically 4–16, versus thousands of text tokens for a scraped web page.

## Why it matters

- **Decouples context length from tool-output verbosity.** A search result returning 5000 tokens becomes 8 latent tokens.
- **No fine-tuning of the agent.** $E_\phi$ is a small model trained independently; the agent LLM is frozen.
- **No lossy reconstruction.** The objective preserves *action behavior*, not surface text, so the latent can drop irrelevant content aggressively.
- Reported gains on web-search QA: **70.01%–89.24% fewer tokens, 24.45%–55.97% lower end-to-end runtime, higher task performance** than the text-tool-output baseline across two model families and four benchmarks.

## Gotchas & tricks

- The latent is **policy-specific** — a Coca-Cola encoder trained against one agent won't transfer to another agent with different attention patterns.
- Logging / auditability is harder when tool outputs aren't strings; keep raw text in a sidecar store for debugging and compliance.
- **Catastrophic failure mode:** if $E_\phi$ never sees a certain output type at training time (e.g. a new tool is added), its latent degrades silently. Online monitoring of action-KL is advised.
- Latent position in the context still needs positional-encoding treatment — treat the $K$ latent tokens as consecutive positions.

## Sources

- Paper: Tools Need Not Speak in Text: Latent Context Compression for Agents — Tsinghua AI Agent Group, 2026 — [arXiv:2611.06500](https://arxiv.org/abs/2611.06500)
