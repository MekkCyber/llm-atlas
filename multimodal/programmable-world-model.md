# Programmable World Model

*Depth — decouple world-state evolution (executable programs over explicit state) from visual observation (a video model rendering that state).*

**TL;DR:** Pure video world models look right frame-to-frame but drift on *state* — object counts, inventories, HP bars, game-state variables — over long interactions. A programmable world model splits the system into two: a **state engine** running an executable program over an explicit, structured state (e.g. 3D oriented bounding boxes with attached attributes), and a **video model** that renders observations conditioned on that state. Programs enforce rules deterministically; the video model handles pixel fidelity. On CombatStateBench this yields 94% count accuracy and 98% state accuracy.

**Prereqs:** none
**Related:** [visual-calibration](visual-calibration.md), [README](README.md)

---

## What it is

Video world models are the class of generative models trained to predict the next frame of an interactive scene given past frames + an action. They can look impressive but *cannot* be trusted to maintain long-horizon state: after 30 seconds the same character may have three different HP counts across frames, an object placed on a table may vanish, or a counter may reset.

The programmable-world-model recipe replaces "one big video model does everything" with a **hybrid**:

- **State engine.** Maintains a global, symbolic state — e.g. a list of state-augmented 3D OBBs, each with attributes (identity, health, count, orientation). Actions execute programs that mutate this state deterministically.
- **Video renderer.** Conditioned on the current state (and past frames) to produce the next visual observation. Now it renders *given* the state instead of implicitly tracking it.

## How it works

The interaction loop:

```
state_0 = init_state(scene)
for t in 0..T:
    action_t = policy(observation_{t-1}, prompt)
    state_t  = program_engine.apply(state_{t-1}, action_t)   # deterministic
    obs_t    = video_model.render(state_t, past_frames)      # learned
```

Concretely:

- **State representation.** Each entity in the scene is a 3D OBB with attached scalar/categorical attributes (health, count, phase).
- **Programs.** Each action type is associated with a program (a small piece of code) that mutates OBBs and attributes. Rules — "when HP hits 0, remove entity" — live in the programs.
- **Rendering conditioning.** OBB positions + attributes are fed to the video model as conditioning tokens; the model learns to render each state consistently.

## Why it matters

- **Correctness where it matters.** If world models are going to serve as policy-in-the-loop simulators for RL post-training (see [../post-training/wmrl.md](../post-training/wmrl.md)), state fidelity is more important than pixel fidelity — a policy trained against an incorrectly-counting simulator is worse than useless.
- **Composable authoring.** Because rules live in programs, they can be edited, extended, or debugged. A pure video model is a black box.
- **Cheap to trust.** The programmable side is deterministic; only rendering is learned. Failure surface shrinks.

## Gotchas & tricks

- **State abstraction is the design problem.** OBB + attributes works for game-like scenes; for open-ended domains (long-form video, real world), the state schema is harder to fix ahead of time.
- **Rendering-state mismatch.** The video model can still hallucinate details not in the state — attributes need to cover the visible-behavior surface.
- **Program complexity.** For rich domains (e.g. crafting, physics) the programs balloon; the elegance of "just render it" is what the paper is trading away.
- **Not the same as a symbolic simulator.** A symbolic simulator would also refuse pixels; the programmable world model still learns visuals — the point is *decoupling* rules from pixels, not eliminating either.

## Sources

- Paper: *Programmable World Model* — Lin, Lin, Huang, Yu, Niu, Yang, Liu, Chuang, Zhang, Wang, 2026 — [arXiv:2609.10540](https://arxiv.org/abs/2609.10540).
