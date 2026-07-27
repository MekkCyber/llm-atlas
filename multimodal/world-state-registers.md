# World State Registers
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A mechanism for maintaining **shared, persistent world state** in autoregressive video diffusion models across multi-agent and multi-view rollouts. Instead of carrying pixel history as conditioning context, the model keeps a compact set of **learnable register tokens** that store global world information plus per-agent status, and updates them after each generated chunk. Introduced in WorldWeaver (W²).

**Prereqs:** [README.md](./README.md), [../architectures/transformer-block.md](../architectures/transformer-block.md).
**Related:** [../architectures/_moe.md](../architectures/_moe.md) · [../case-studies/composer2.md](../case-studies/composer2.md)

---

## What it is

The setting: multi-agent interactive world models (e.g. two-agent Minecraft video generation) need to generate *consistent* observations across agents and *evolving* world state across views. Standard autoregressive video diffusion carries observation history as conditioning context, which:

- Grows quadratically with the number of agents × views.
- Makes it hard to represent state that isn't visible in any single frame (agent inventories, distant objects, scene-wide events).

World state registers replace this with a bounded, structured state — learnable tokens that live alongside the visual tokens in the diffusion transformer and are read/updated at every rollout step.

## How it works

- **Register bank.** A set of $K$ learnable tokens per rollout, partitioned into "global" and "per-agent" slots.
- **Read.** Each generation step, the diffusion transformer cross-attends over the register bank as part of its conditioning (alongside the current noisy frame tokens).
- **Update.** After each generated chunk, a lightweight update head produces new register values from the chunk's latent — the registers are the residue passed to the next step.
- **Grounding supervision.** During training, registers are supervised with auxiliary signals: individual agent status, global-state views (bird's-eye), scene text. This forces the model to *actually use* the register slots to store the relevant state, rather than treating them as free-form scratch.

WorldWeaver pairs the registers with a **Mixture-of-Transformers** design: distinct weight blocks for world-state modeling vs visual frame modeling, so the register update dynamics don't compete with visual generation quality.

## Why it matters

Video world models are the substrate for both interactive entertainment and for **training environments** for VLA / embodied agents. Shared state across agents is a hard requirement for either — you can't have two agents cooperate if their observations are independently hallucinated. Registers make shared state a first-class object of the diffusion pipeline, and the grounding supervision makes it *trainable* rather than emergent.

The pattern generalizes: any autoregressive generative model that needs persistent latent state across steps can use the register-bank + supervised-update recipe.

## Gotchas & tricks

- **Register count is a hyperparameter.** Too few and the model can't hold the world; too many and the register update becomes as expensive as the frame generation.
- **Supervision is load-bearing.** Without auxiliary supervision on the register contents, registers tend to collapse to task-irrelevant scratch. BEV / scene-text supervision is what makes them behave like state.
- **Not a memory in the LSTM sense.** Registers don't have a recurrent state equation — they're just tokens updated by a learned head. All the "memory" is in the transformer's attention over them.
- **MoT vs registers.** The Mixture-of-Transformers wrapper (separate weights for state vs visual modeling) is orthogonal — you can use registers with a monolithic transformer, or MoT without registers. WorldWeaver combines both.

## Sources

- Paper: *Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers* — Sicheng Mo, Yuheng Li, Ziyang Leng, Krishna Kumar Singh, Bolei Zhou — UCLA / Adobe Research, 2026 — [arXiv:2607.21594](https://arxiv.org/abs/2607.21594)
- Project page: https://vail-ucla.github.io/worldweaver/
