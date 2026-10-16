# Agentic Video Generation (ATAS)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Agent-loop pattern for long-horizon video generation. A **planner** decomposes a long prompt into shots (camera moves + scene edits), a **diffusion generator** renders each shot conditioned on a persistent world state, and the state is updated after each shot so later shots stay geometrically and semantically consistent. The world state tracks both **when** (time) and **where** (space).

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** [README.md](./README.md)

---

## What it is

Text-to-video models drift as horizons grow. Identities, geometry, and camera pose decay over seconds-to-minutes of generated video because the generator has only the input prompt and the previous few frames — no durable memory of the world being filmed.

**ATAS (Any Time, Any Space)** reframes long-horizon generation as an **agentic loop over a persistent world state**: pixels are what the generator emits, but *the world state* is what persists across shots and across the agent loop.

## How it works

### World state

Structured representation of the scene being filmed. Includes:
- **Time** — the current time in the generated sequence; event history (what happened in prior shots).
- **Space** — object inventory, their positions, camera pose, lighting, scene layout.

Representation is modular — the paper's design keeps the state readable/writable by the planner (text / structured fields) and queryable by the generator (as conditioning).

### Planner

An LLM-style agent that takes the long-horizon user prompt and produces a shot plan: a sequence of (camera-move, scene-event) tuples with references to the world state. The planner reads the current state before each shot and writes updates after each shot.

### Diffusion generator

Conditioned on the current shot's prompt *plus* the world-state slice relevant to this shot. Produces the video frames for one shot. The output includes updates fed back into the world state (object positions, scene changes).

### Loop

```
Loop over shots:
    state_slice = planner.read_state(world_state)
    shot_prompt = planner.plan_shot(user_prompt, state_slice)
    frames      = generator.render(shot_prompt, state_slice)
    world_state = planner.update_state(world_state, frames, shot_prompt)
```

Later shots can reference earlier events ("the cat from shot 2 is now on the sofa") because the state carries that information explicitly.

## Why it matters

- **Agent memory for generative media.** Agent-loop patterns from LLM tool use (plan, act, observe, update memory) transfer to video generation. The world state is to video what a scratchpad is to an LLM agent.
- **Breaks the "context window" bottleneck for video.** Instead of concatenating earlier frames into attention, the state is a compressed, structured memory. Scales to minutes-long sequences in principle.
- **Decouples planning from pixel generation.** Each shot can use a strong (frame-stable) short-horizon diffusion generator; long-horizon consistency is handled by the planner + state, not by a bigger generator.
- **Signpost paper.** Combines two previously separate research threads (LLM agents, video diffusion) under one abstraction.

## Gotchas & tricks

- **State fidelity.** The world state is only as good as the planner's updates. Sloppy state updates compound.
- **Generator-state binding.** How the diffusion generator uses the state (as text, as structured conditioning, as a learned encoding) is a design axis; different bindings have different sensitivity to state inaccuracies.
- **Not an end-to-end objective.** The system is modular; no single loss trains the planner + generator + state together. Each component is trained/prompted separately.
- **Prompt-grounded planning can hallucinate.** The planner must stay grounded in the user prompt; otherwise the shot plan drifts from the user's intent.
- **Evaluation is open.** Long-horizon video consistency metrics are an active area; existing short-clip benchmarks don't test what this method is for.

## Sources

- Paper: *Any Time, Any Space: A World-State-Centric Framework for Agentic Video Generation* — Wang, Zhang, Li, Liu, Zhang, Zhao, He, Chen, Chen, Zhuang (Zhejiang / Monash), 2026 — [arXiv:2610.20571](https://arxiv.org/abs/2610.20571). Method name: ATAS.
