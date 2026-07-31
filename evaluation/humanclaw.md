# HumanCLAW — Decoupled-Execution Benchmark for Embodied VLMs

*Depth — an evaluation framework that measures VLM decision-making by factoring out low-level motor noise.*

**TL;DR:** Embodied-VLM benchmarks confound two failures: the model made a bad decision, or the motor controller failed to execute a fine decision (lost balance, missed grasp, hit an obstacle). HumanCLAW gives the VLM a fixed atomic-skill vocabulary; each skill is translated by a physics-aware full-body controller into a sub-second motion chunk with real gravity and collisions. What the benchmark scores is the VLM's *choice at every step*, not the controller's tracking. Bench: 1,218 long-horizon egocentric find-navigate-interact episodes across 41 indoor scenes.

**Prereqs:** *(none — introduces its own evaluation methodology)*
**Related:** [../multimodal/README.md](../multimodal/README.md), [../multimodal/_vla.md](../multimodal/_vla.md), [../agents/README.md](../agents/README.md)

---

## What it is

An evaluation setup for embodied VLMs that treats decision-making and low-level control as orthogonal. The VLM sits at the top of a two-layer stack:

- **Decision layer (evaluated):** the VLM sees egocentric frames + a task instruction, emits one atomic skill command per step ("turn 30° left", "pick up the mug", "step forward 40 cm").
- **Execution layer (fixed):** a harnessed full-body controller translates the command into a physics-simulated motion chunk with real body dynamics.

Any failure is attributed to the decision layer *by construction* — the controller is held constant across models.

## How it works

- **Skill vocabulary.** A small closed set of atomic commands that a strong controller can execute reliably. Restricting to atomic skills is what keeps the execution layer's failure rate low.
- **Physics rollout.** Each skill produces sub-second motion in a full-body physics simulator (gravity, collisions). The body can lose balance, hit walls, or miss objects; these physical consequences propagate to the next frame the VLM sees.
- **Long-horizon episodes.** 1,218 find-navigate-interact tasks in 41 indoor scenes; success requires tens to hundreds of correct decisions.
- **Metric.** Episode-level success rate. Nine SoTA VLMs evaluated; the best reaches **16.8%** success — nowhere near solved.

## Why it matters

- **Reveals what perception benchmarks hide.** Standard VLM benchmarks measure recognition. HumanCLAW shows recognition isn't the bottleneck for embodied use; **embodied self-awareness** is — models lose track of their own body location, don't reliably know when the goal is reached, don't notice obstacle contact.
- **Transferable evaluation pattern.** The decouple-decision-from-execution shape is reusable anywhere a strong controller exists (drones, humanoids, mobile manipulators, dexterous hands).
- **Anchors the [`_vla`](../multimodal/_vla.md) taxonomy's skill-vocabulary branch.** Shows the strengths and current ceiling of that VLA design point.

## Gotchas & tricks

- **Skill vocabulary is the ceiling.** A task that requires an action outside the skill set is unsolvable by definition, regardless of decision quality. Benchmark authors have to co-design the skill set with the task distribution; readers should check whether a low score reflects decision failure or vocabulary starvation.
- **Not a general VLA benchmark.** Because the execution layer is fixed, HumanCLAW does **not** measure joint end-to-end policies (like [TurboVLA-style](../multimodal/_vla.md) V+L→A). It's a decision-layer benchmark, and its numbers aren't directly comparable to end-to-end VLA success rates.
- **Body-tracking sensors matter.** The reported "loses track of its own body" failure is measured in a simulator; a deployed system with proprioception sensors might not have exactly this failure mode — the finding is about *what current VLMs infer from egocentric frames alone*.
- **Judge / eval reliability.** Success is defined per episode (goal reached, no collision violation). Long-horizon episodes have many ways to trivially fail early; look at per-stage completion, not just terminal success, to compare models fairly.

## Sources

- Paper: *HumanCLAW: Can Vision-Language Models Act Through a Body?* — Gu et al., Meta / NTU / UW / Brown / Northwestern, 2026 — introduces the benchmark and the decoupled-execution methodology. See [../daily-papers/2026-07-30.md](../daily-papers/2026-07-30.md).
- Related: [_vla.md](../multimodal/_vla.md) for the VLA design space HumanCLAW's scoring targets.
