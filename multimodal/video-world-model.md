# Video world model (trajectory-decoupled)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Existing video "world models" entangle physical dynamics with pixel rendering — they need continuous visual observation to preserve object identity, and any occlusion breaks the model. A **decoupled** world model separates **semantic motion orchestration** (an LLM plans 3D trajectories + camera moves) from **visual generation** (a video generator renders conditioned on those trajectories). Persistent object memory lives in the trajectory, so entities that leave the scene and return keep their exact visual identity. Introduced by WorldDirector.

**Prereqs:** [attention](../fundamentals/attention.md), [README](README.md)
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

A **video world model** predicts future frames conditional on actions or agent controls — the base primitive for embodied agent training in simulation, game engines with generative frames, and interactive video.

The **pixel-coupled** design (dominant since Genie / DIAMOND-style world models) trains a single generator to predict next frames from history. Object identity is only preserved for as long as objects stay visible. Once occluded — behind another object, off-screen — the model has to *reinfer* their identity when they reappear, and drift is nearly universal.

The **trajectory-decoupled** design (WorldDirector, 2026) splits the problem in two:

- **Planner (LLM).** Maintains persistent 3D trajectories for every dynamic entity and a camera pose. When the user requests a viewpoint change or the scene evolves, the LLM updates trajectories using a symbolic representation of the scene state.
- **Generator (video diffusion / flow).** Renders frames conditioned on the current trajectories and camera pose — including for entities that were off-screen and are now returning.

Object identity is stored in the trajectory, not in the pixels, so persistence is a *plan-time* property.

---

## How it works

**Trajectory state.** Per dynamic entity: a 3D bounding trajectory, an identity token, and a compact appearance descriptor. Trajectories live in world coordinates independent of camera.

**LLM planner.** Given the current state and a user goal ("move to the other side of the room"), the LLM emits updated trajectories and a camera path. Physical logic (occlusion order, collision, re-entry) is a language-level plan, not a pixel-level generation step.

**Video generator.** A conditional diffusion / flow-matching video model that takes trajectories + camera as control signals (e.g. via cross-attention or explicit projection layers) and renders frames. Because trajectories carry identity tokens and appearance descriptors, re-entering entities are rendered consistently with their prior appearance.

**Long-horizon control.** Unrestricted viewpoint changes and prolonged occlusion don't drift, because the pixel model never has to *remember* what an entity looked like — it always has the trajectory + descriptor.

---

## Why it matters

- **Persistent identity.** Solves the drift-on-occlusion failure mode that has been the defining limitation of pixel-coupled video world models.
- **Free-viewpoint exploration.** Because the trajectory / camera split is explicit, cameras can be moved arbitrarily without re-training or per-camera fine-tuning.
- **Composable with agents.** The LLM planner is the same abstraction agents already use for text-domain tool use — the world model just adds "video generation" as one of the tools.
- **Interpretable failures.** When something goes wrong, you can inspect the trajectory / camera plan separately from the render — a large debugging win over end-to-end pixel models.

---

## Gotchas & tricks

- **LLM planning cost.** Every frame or key-frame requires an LLM inference. Cache aggressively; only re-plan when scene state changes non-trivially.
- **Trajectory ↔ generator interface is load-bearing.** Poor conditioning (e.g. dumping raw trajectories into a text prompt) undoes the identity-persistence benefit. WorldDirector uses structured control signals aligned with the generator's conditioning path.
- **Not free from generator artifacts.** The generator still has to render coherent frames; long-horizon quality still depends on video-model capacity.
- **Appearance descriptors are their own design space.** Too coarse (one embedding) collapses identity; too fine (full 3D asset) is expensive. WorldDirector uses a compact per-entity descriptor bound to the identity token.
- **Doesn't handle novel-object generation gracefully.** Persistent identity assumes the entity was seen at least once. Bringing in a completely new object mid-episode is still a hard case.

---

## Sources

- Paper: *WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory* — Wang et al., 2026 — [arXiv:2607.02517](https://arxiv.org/abs/2607.02517).
- Precursors: pixel-coupled world models — Genie, DIAMOND, Sora-style rollouts (referenced in paper §2).
