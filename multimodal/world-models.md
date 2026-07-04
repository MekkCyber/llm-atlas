# Video World Models
*Depth — generative video models used as environment simulators, with an emphasis on decoupled semantic/visual state.*

**TL;DR:** A **video world model** takes a caption or state description and generates video that *simulates* an environment — objects with persistent identity, physical dynamics, controllable camera. The hard part is *persistence*: when an object leaves frame and returns, most video generators lose its identity because state is entangled with pixels. **WorldDirector** shows one way out — an LLM plans 3D object trajectories and camera in symbolic space, and a video generator renders whatever trajectories it's handed. Semantic memory lives in the planner; the renderer is stateless.

**Prereqs:** (no diffusion / video depth pages yet in the graph)
**Related:** [../agents/memory-as-skill.md](../agents/memory-as-skill.md)

---

## What it is

Two flavors of video world model:

- **End-to-end.** A single video generator conditioned on action / caption history. State is implicit in the model's latent activations; dynamics and rendering are entangled. When the model must recall an object after a long occlusion, it either hallucinates a new one or drifts identity.
- **Decoupled.** Separate the **semantic planner** (what should happen: object positions, trajectories, camera moves) from the **visual renderer** (what it should look like: video generator conditioned on trajectory controls). Persistent state lives in the planner as symbolic data; the renderer stays memory-less.

WorldDirector is a canonical decoupled instantiation. An LLM coordinates 3D trajectories with camera movements — object A crosses the room from (x₁, y₁, z₁) to (x₂, y₂, z₂), camera pans, object A re-enters at (x₃, y₃, z₃) with its exact original appearance — and passes these to the video model as control signals.

## How it works

1. **Scene state.** Objects are represented symbolically (id, appearance embedding, current 3D pose, physical properties). Not pixels.
2. **LLM director.** Given a task or script, the LLM plans a sequence of state updates: object trajectories, camera poses, event triggers. It maintains identity because it's operating on structured symbols, not video frames.
3. **Trajectory-to-video renderer.** A video generation model (typically a text-to-video diffusion) is conditioned on the LLM-planned trajectories (via ControlNet-like signals or camera/motion tokens) and renders each shot.
4. **Persistence via re-injection.** When object A re-enters after occlusion, its stored appearance embedding is fed back into the renderer as identity conditioning — the renderer gets its identity from the state, not from having "remembered" it.

## Why it matters

- **Persistent identity across long horizons.** End-to-end video models drift; decoupled world models keep identity because it's tracked in symbolic state, not pixels.
- **Unrestricted viewpoint exploration.** Since the world state is 3D-aware and separate from the render, arbitrary new camera trajectories can be rendered from the same underlying state.
- **Divide of labor mirrors LLM+image-generator systems.** The pattern of "language model plans, generator renders" that works for text-to-image (e.g. code-driven image generation, layout-controlled diffusion) transfers to video with world-state added.

## Gotchas & tricks

- **Identity conditioning is fragile.** Appearance embeddings drift across long shots unless re-injected. Frequent re-conditioning keeps identity stable but constrains the renderer.
- **Physics is only as good as the LLM.** Symbolic trajectories from the LLM aren't guaranteed physically consistent; augment with a physics prior or verifier if realism matters.
- **The LLM cost is real.** For long or complex scenes, LLM planning dominates latency; batch and cache trajectory plans.
- **Related to text-to-3D-scene systems** — many share the "LLM plans, diffusion renders" division but for static scenes; video world models add temporal dynamics.

## Sources

- Paper: *WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory* — Wang et al., 2026 — [arXiv:2607.02517](https://arxiv.org/abs/2607.02517).
- Related: *Genie / Sora / VideoPoet* — end-to-end video world models this contrasts with.
- Related: *DirectorLLM / VideoDirectorGPT* — earlier LLM-plans-video work in the same lineage.
