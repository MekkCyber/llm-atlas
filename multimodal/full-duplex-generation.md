# Full-Duplex Audiovisual Generation
*Depth — chunked autoregressive reasoning synchronized with a diffusion-style audiovisual generator for real-time, interruptible multimodal interaction.*

**TL;DR:** Most multimodal LLM systems are turn-based: user speaks, model listens, then model replies. **Full-duplex** means the model perceives and generates simultaneously, can be interrupted, and decides for itself when to participate. The Duo v1 "Reciprocal World Model" formulation splits that into two modules on a shared chunked timeline — an autoregressive **Intent** module (cognition and participation decisions) and a diffusion-style **Realization** module (audiovisual output) — trained with **dual rollout distillation** so each module is robust to the other's generated history.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md)

---

## What it is

A two-module multimodal system that runs continuously while the user interacts:

- **Intent:** an autoregressive LLM-style head that at each chunk decides what the agent should think, whether it should speak now, and what high-level action to take. Chain-of-thought and participation decisions live here.
- **Realization:** a diffusion-style generator that produces the *audiovisual* output for each chunk — the actual voice waveform and video frames. Conditioned on Intent's decisions.

Both modules consume real-time audio, video, and text inputs chunk by chunk. Both modules can decide to be silent, speak over, or yield. That's what "full-duplex" means here.

## How it works

### Shared chunked timeline

Time is discretized into fixed chunks (e.g. 200 ms). At chunk $t$:

- Intent token stream: $\mathrm{ints}_{\le t}$ — Intent's autoregressive outputs so far.
- Realization token stream: $\mathrm{real}_{\le t}$ — audiovisual chunks so far.
- Observation stream: $\mathrm{obs}_{\le t}$ — user-side audio/video/text chunks so far.

Intent and Realization **share synchronized indices** so chunk-$t$ outputs correspond in wall-clock time. Per-module **causal masks** enforce (1) both can see all observations up to $t$, (2) Intent cannot see future Realization or vice versa, (3) neither sees future observations.

### Dual rollout distillation

The core training trick. Autoregressive modules suffer *exposure bias* — they're trained on ground-truth history but roll out on their own generated history, which drifts. Dual rollout distillation closes the gap by training each module on the *other module's* autoregressive output:

- Train Intent on Realization's generated audiovisual chunks (as observation).
- Train Realization on Intent's generated intent tokens (as conditioning).

Both modules learn to be stable under the realistic distribution they'll see at inference, not the clean ground-truth distribution.

## Why it matters

- Turn-based multimodal agents can't stand in for a conversation partner: they don't interrupt, they don't yield, and they can't respond during the user's turn. Full-duplex is the architectural shape that fixes that.
- Separating cognition (Intent, AR) from perception+action (Realization, diffusion) lets each module use its strongest modeling regime instead of forcing one architecture to do both.
- The dual-rollout trick generalizes to any multi-module AR system that produces conditioning for another AR module — it's the same shape as speculative decoding's draft/target asymmetry.

## Gotchas & tricks

- Chunk granularity sets latency floor and per-chunk compute. Too coarse → unresponsive; too fine → every chunk wastes transformer prefill overhead.
- Per-module causal masks must be correct down to the token — a single leak of future Realization into Intent lets the model cheat on its own participation decision.
- Realization's output distribution is wide (audio + video jointly), so without rollout distillation its behavior at deployment diverges fast from training.

## Sources

- Paper: *Duo: A Reciprocal World Model for Full-Duplex Interaction* — 2026 — https://arxiv.org/abs/2611.15131
