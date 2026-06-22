# Persistent-State World Models
*Depth — the requirement that video world models maintain world state under occlusion and re-observation.*

**TL;DR:** A **world model** isn't a renderer — it should maintain the **evolving state of the world** even when the camera looks away, and reproduce that state on re-observation. Most current video generation models fail this test: they render plausible motion but **don't keep state**. The "preservation–access–re-observed-consistency gap" was named and quantified by *Current World Models Lack a Persistent State Core* (Lu et al., 2026) on WRBench. The fix the paper argues for: explicit **"what-memory" mechanisms** and **endpoint-persistence training objectives**.

**Prereqs:** [README.md](README.md)
**Related:** [../evaluation/wrbench.md](../evaluation/wrbench.md), [image-editing-wam.md](image-editing-wam.md)

---

## What it is

Define a **world model** $f$ as a function that, given an observation history and a controllable input (camera path, action), predicts future observations consistent with an underlying world state $s$:

$$
\hat{o}_{t+1} = f(o_{1:t}, \text{control}_{t+1}; s)
$$

For $f$ to be a real world model rather than a "video renderer," it must satisfy three properties when the camera leaves and returns to a vantage:

| Property | Definition |
| --- | --- |
| **Preservation** | Events that change $s$ off-camera are encoded in latent state. |
| **Access** | $s$ is retrievable from latent state when needed (re-observation triggers retrieval). |
| **Re-observed consistency** | The re-rendered scene reflects the *post-mutation* $s$, regardless of camera path. |

Current video generation models — DiTs trained on next-token-style next-frame objectives — typically satisfy none of these. They satisfy a weaker property: per-frame perceptual plausibility.

## How it works (in models that fail)

A standard video DiT generates frame $t+1$ from a context window of prior frames $o_{t-k:t}$ via attention. The world state $s$ is only ever represented implicitly in the model's hidden activations at the moment those activations are computed. When the camera moves away and the relevant entity exits the context window, **the model has no place to keep its state** — context attention can't see it, and there's no separate world-state memory.

Three failure modes follow naturally:

1. **Preservation failure** — the off-camera event isn't encoded. Re-observation re-rolls the state.
2. **Access failure** — the event is encoded somewhere but the model can't retrieve it.
3. **In-place vs relocation asymmetry** — relocations get a free hint from object-presence cues; in-place state changes (cup filled in the same location) have no such hint and fail harder.

## How it works (what's needed)

The paper sketches the requirements without prescribing an architecture:

- **What-memory module.** A persistent store keyed by entity identity, written to whenever an event changes the entity's state.
- **Endpoint-persistence objective.** A training loss that explicitly checks consistency at re-observation, not just per-frame fidelity. Candidates: contrastive loss between actual and re-rendered re-observations, masked-state reconstruction objectives.
- **Camera-trajectory diversity.** Training on trajectories that include leave-and-return motion, not just panning / orbiting.

## Why it matters

- **Reframes "world model" as a memory problem**, not a rendering problem. Most existing video pretraining doesn't address it.
- Reveals that fields counting on video pretraining as a *substitute* for real world modeling (embodied AI, video Q&A, physical reasoning) are sitting on a structural gap.
- Motivates alternative WAM designs that **sidestep** video prediction entirely (see [image-editing-wam](image-editing-wam.md)), or augment the video model with explicit memory.

## Gotchas & tricks

- **Better video metrics don't fix this.** FVD / CLIPScore are unable to detect persistence failures. Use [WRBench](../evaluation/wrbench.md) or a similar probe.
- **In-place state changes are the diagnostic.** A model strong on relocations but weak on in-place changes is exploiting object-presence cues, not modeling state.
- **Adding context length helps marginally.** Doubling the context window improves preservation in short scenarios but doesn't fix the structural gap — past a few thousand frames, even very long contexts don't substitute for a dedicated state store.
- **External memory experiments are open.** Retrieval-augmented video models and entity-keyed memory are early; nothing standard yet.

## Sources

- Paper: *Current World Models Lack a Persistent State Core* — Lu, Zhu, Shi, Cai, Tang, Chen, Cao, Tang, Zhang, Dai, Ju (USTC, X-Humanoid, CAS, PKU), 2026, arXiv 2606.20545 — the framing and WRBench results.
- Related: *ImageWAM* — [image-editing-wam](image-editing-wam.md), 2026 — alternative design that sidesteps the problem.
