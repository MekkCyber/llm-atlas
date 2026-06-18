# GameCraft-Bench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A 140-task benchmark that asks coding agents to build complete, playable Godot games from a natural-language spec and grades the result by *actually playing* the game through replayed demonstrations scored by a rubric-guided multimodal judge. Frontier coding agents top out at 41.46% — game generation is much harder than text-only coding tasks.

**Prereqs:** [livecodebench](livecodebench.md)
**Related:** [humaneval](humaneval.md), [../agents/README.md](../agents/README.md)

---

## What it is

Most coding-agent benchmarks (HumanEval, LiveCodeBench, SWE-bench) measure patches against a fixed text corpus: the agent writes code, the harness compiles or unit-tests it. **End-to-end game generation** breaks that contract — the artifact is not just source code but **scripts + scenes + assets + rendering + runtime interactions**, all of which must compose into something a user can *play*.

GameCraft-Bench formalizes this as a benchmark with three desiderata:

- **Engine Grounding.** The artifact must run in a real game engine (Godot), not a stub.
- **Artifact Completeness.** Scripts, scenes, and assets are all required; partial implementations fail.
- **Interactive Verification.** Grading happens against a **replayed playthrough**, not against the source code.

---

## How it works

### Task structure

140 tasks across 15 game families (platformer, puzzle, top-down RPG, top-down shooter, racing, etc.). Each task gives the agent:

- A natural-language game spec.
- The Godot environment.
- Permitted asset / library access.

The agent must produce a complete Godot project that, when run, exhibits the specified mechanics, content, and presentation.

### Replay-based judging

Each task ships with a set of **replay demonstrations** — scripted player inputs that exercise the intended mechanics. To grade an agent's submission:

1. Run the agent's project in Godot.
2. Apply the replay inputs.
3. Record the gameplay.
4. Run the recording through a rubric-guided **multimodal judge** that scores along axes like mechanic correctness, visual feedback, content completeness, and presentation coherence.

The judge sees what a player would see, not the source code — closing the loop that text-only judges miss (e.g., a game that compiles but renders nothing).

### Three failure axes

The rubric explicitly separates:
- **Mechanics** (does the game actually do what was asked?)
- **Visual feedback** (does the player see meaningful state changes?)
- **Presentation coherence** (do the pieces look like one game?)

Most agents pass the first axis (recognizable mechanics) but fail the second and third — a fingerprint of LLM-built games circa 2026.

---

## Why it matters

- **First benchmark to grade an agent's output by playing it.** SWE-bench grades patches against unit tests; GameCraft-Bench grades the *experience*. That's a much harder integration test that exposes how far current agents are from end-to-end product delivery.
- **Frontier agents at 41.46% headroom.** A clear improvement axis exists for coding agents that target non-text artifacts.
- **Reusable judging methodology.** Replay-based multimodal judging applies anywhere the artifact has dynamic visual behavior — game generation today, broader interactive-artifact benchmarks tomorrow.

---

## Gotchas & tricks

- **Multimodal judge is a model.** The rubric reduces variance, but the judge is still an LLM scoring a recording; calibration / bias studies matter.
- **Replay inputs constrain what's testable.** Designed-by-hand replays test the intended mechanics; emergent behavior (player creativity) isn't measured. Treat scores as floors, not ceilings.
- **Godot-only.** Engine grounding is real, but Godot-specific patterns (GDScript, Godot's scene system) bleed into the artifact distribution. Cross-engine generalization is an open question.
- **Asset generation conflation.** Some tasks bundle code + asset generation; failures can be attributed to either side. Agents that fail visual-feedback metrics often have working scripts but missing or broken assets.

---

## Sources

- Paper: *GameCraft-Bench: Can Agents Build Playable Games End-to-End in a Real Game Engine?* — Rongsheng Wang et al., CUHK-Shenzhen / Tencent Hunyuan et al., 2026 — [arXiv:2606.17861](https://arxiv.org/abs/2606.17861).
