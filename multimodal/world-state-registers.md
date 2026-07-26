# World State Registers

*Depth — persistent learnable tokens that carry shared world facts across agents and views in multi-agent video diffusion.*

**TL;DR:** Multi-agent interactive world models need to keep shared state consistent across views and long horizons, but standard autoregressive video diffusion carries state implicitly in the observation history — brittle and drift-prone. **World state registers** are a small set of learnable tokens that persist across generated chunks, updated after each chunk, and supervised to encode shared world facts (global bird's-eye views, scene text) plus per-agent status. Combined with a Mixture-of-Transformers split (separate weights for state modeling vs. visual frame modeling), they let two-agent Minecraft rollouts stay logically consistent that would otherwise diverge. Introduced by **WorldWeaver (W²)** (UCLA & Adobe, 2026).

**Prereqs:** [README.md](./README.md), [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md)

---

## What it is

Streaming multi-agent video diffusion produces a sequence of chunks $x_0, x_1, \ldots$ per agent, conditioned on some shared world state. Two failure modes have dominated:

1. **State drift.** With state carried implicitly in the observation history conditioning, small per-chunk errors compound; by chunk 20 the two agents are effectively in different worlds.
2. **View inconsistency.** Agent A sees a red door; agent B, generated separately, generates a blue one. Any shared fact must be re-derived from overlapping observations.

Registers make the shared state an *explicit, updateable object* — a fixed-shape set of learnable tokens the model reads from and writes to, in the style of a working memory.

## How it works

Three moving parts:

1. **Register tokens.** A set of $K$ learnable vectors (e.g. $K = 32$) attached to every chunk-generation forward pass. Each token has a semantic role: shared world state, per-agent status, or a supervised bird's-eye / scene-text summary.
2. **Register update.** After each generated chunk, the registers are re-computed from the previous registers + the chunk's produced content. Update is a learned function (attention over chunk tokens with registers as queries) that runs once per chunk.
3. **Mixture-of-Transformers split.** Separate weights for *world-state modeling* (produces & consumes registers) and *visual frame modeling* (produces & consumes pixels), coordinated through the shared register set. This prevents state gradients from fighting pixel gradients — a form of coarse-grained MoE at the block level.

Supervision signals ground the registers so they don't drift into arbitrary latent codes: bird's-eye-view predictions, scene-text extraction, per-agent status labels.

## Why it matters

- **Explicit state is diffable state.** Registers can be logged, edited, or reset without regenerating video. Debugging multi-agent inconsistencies goes from "roll back and hope" to "inspect the register at chunk 12."
- **Compresses long-horizon consistency into constant-size memory.** Observation-history conditioning grows with time; registers stay fixed-shape, decoupling horizon length from context cost.
- **A transferable primitive.** The same register pattern is a plausible substrate for embodied agents (shared inventory / map state), game AI (world facts persistent across NPCs), and multi-agent simulators.
- **MoT split isolates state gradients from pixel gradients.** A recurring finding across recent multimodal architectures: modality-specific weights outperform shared weights when the modalities have very different learning dynamics.

## Gotchas & tricks

- **Register count vs update rate.** More registers = more capacity but slower update; higher update frequency = fresher state but higher cost per chunk. The paper fixes both; downstream users will retune.
- **Registers can collapse.** Without grounding supervision they can degenerate into a learned prior on the training distribution. The bird's-eye / scene-text supervision is not decorative — it's what keeps registers meaningful.
- **Multi-agent means multi-writer.** If two agents update the same register concurrently, ordering matters. WorldWeaver serialises updates per generation step; alternative concurrent-update schemes remain open research.
- **Registers ≠ KV cache.** KV caches store *observed* tokens verbatim; registers store a *learned* compression of them. Confusing the two produces the wrong scaling intuition.
- **Portability to language / other modalities.** The specific MoT split and supervision signals are video-specific. The abstract pattern (persistent learnable-token memory updated per chunk) is more general and worth trying for streaming any-to-any and long-agent-trajectory settings.

## Sources

- Paper: *Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers* — Mo, Li, Leng, Singh, Zhou — UCLA & Adobe Research, 2026 — introduces WorldWeaver / W², world state registers, and the MoT split for streaming multi-agent video.
- Related lineage: memory-augmented transformers (e.g. Memorizing Transformers), Perceiver-style latent bottlenecks, and working-memory tokens in RL agents.
