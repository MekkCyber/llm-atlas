# World State Registers
*Depth — a fixed-size bank of learnable tokens that stores and updates shared world state across chunks in multi-agent video diffusion.*

**TL;DR:** Autoregressive video diffusion normally carries observation history in its conditioning context. In multi-agent or multi-view scenes that's brittle: shared world state (positions, resources, other agents' status) gets lost as the context ages. **World state registers** are a fixed-size set of learnable tokens that live *outside* the observation stream, are prompted into every chunk generation, and are *updated* after each chunk from grounded auxiliary supervision (per-agent status, bird's-eye views, scene text). They decouple world memory from context length.

**Prereqs:** [../multimodal/README](README.md), [../architectures/transformer-block](../architectures/transformer-block.md)
**Related:** [../architectures/mixture-of-transformers](../architectures/mixture-of-transformers.md)

---

## What it is

A register bank $R \in \mathbb{R}^{K \times d}$ of $K$ learnable tokens carried alongside the visual token stream in a video diffusion transformer. During per-chunk denoising, register tokens attend to and are attended by visual tokens, so they influence generation and are influenced by the current chunk's content. Between chunks, an update step rewrites the registers using auxiliary supervision so the next chunk's generation starts from an updated state.

## How it works

Two loops:

**Per-chunk denoising loop (standard diffusion, extended):**
- Concatenate visual tokens with the current register tokens.
- Attention layers mix both freely.
- Visual tokens are denoised as usual; register tokens are held (their gradient comes through auxiliary losses).

**Between-chunk update loop:**
- After a chunk is generated, extract grounding signals: per-agent status, bird's-eye view of the shared world, on-screen text.
- Fold those signals into the registers via a small update module (often a small transformer whose queries are the registers and whose keys/values are the grounding signals).
- Roll the updated registers forward as the next chunk's conditioning.

Register tokens are typed: some track *global state*, others *per-agent status*, others *scene semantics*. Auxiliary losses supervise each register subset against the appropriate signal so the register roles remain disentangled.

Backbone is usefully a **Mixture-of-Transformers**: separate weights for world-state modeling and for visual-frame modeling, since the two token classes prefer very different capacities and gradients.

## Why it matters

Context-carry pipelines conflate *observation history* with *shared world state*. In a solo-agent single-view setting they are almost the same; in multi-agent settings they diverge — one agent sees things another doesn't, and the "shared world" is nowhere in either agent's frame. Registers give the model a dedicated substrate for the shared part, updated from grounded signals rather than inferred from the pixel stream.

## Gotchas & tricks

- **Register count matters.** Too few and the world collapses to a summary; too many and updates become noisy. Two-agent Minecraft experiments landed in the low tens.
- **Auxiliary supervision is not optional.** Without grounded update signals, registers drift into whatever "helps denoising" — usually redundant with visual tokens.
- **Type the registers.** Undifferentiated registers mode-collapse; typing (global / per-agent / scene text) preserves separable meaning.
- **Register-visual token attention should be dense.** Sparse or gated attention between the two token classes hurts state propagation in short chunks.

## Sources

- Paper: *Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers* — Mo, Li, Leng, Singh, Zhou (UCLA & Adobe), 2026 — [arXiv:2607.21594](https://arxiv.org/abs/2607.21594).
