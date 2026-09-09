# Executable Skill Interface
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A contract for VLA / agent skills that treats every skill decision as an **execution proposal**: the runtime checks the skill's prerequisites before executing, invokes the bounded low-level policy, and then verifies the outcome. Because the interface is fixed, the underlying VLA can be replaced without touching the planner. The interface also emits structured planning/execution/verification/recovery trajectories usable as supervision.

**Prereqs:** [agents/README](./README.md), [../multimodal/README](../multimodal/README.md)
**Related:** [../post-training/rejection-sampling](../post-training/rejection-sampling.md), [../post-training/reasoning/prm](../post-training/reasoning/prm.md)

---

## What it is

Long-horizon embodied tasks demand more than action prediction — they need coordination of perception, planning, execution, progress verification, and recovery as the physical state evolves. A bare VLA policy predicts actions but does not, by itself, guarantee that a proposed skill is *valid in the current state* or that its outcome will be *verified*.

The executable-skill interface wraps every skill call `f(state, args) → action` in three phases:

| Phase | What it does | Failure handling |
| --- | --- | --- |
| **Prerequisite check** | Runs a lightweight test that the current state satisfies the skill's declared preconditions | Skill blocked; planner replans |
| **Bounded VLA execution** | Invokes a task-adapted low-level VLA policy with a hard step / time bound | Time-out returns "not verified" |
| **Post-condition verification** | Checks the resulting state against the skill's declared postconditions | Recovery skill scheduled; event logged |

All four events (planning, execution, verification, recovery) are recorded as structured trajectories that flow back into training data for the individual components and — optionally — for online adaptation.

## How it works

The framework instantiates the interface as a single agent loop. High-level skill selection is done by a VLM (e.g. Qwen3-VL); low-level VLA execution runs task-adapted OpenPI / π₀.₅ policies. The interface stays fixed so:

- The planner and low-level policy are decoupled — swap in a different low-level VLA, keep the same loop.
- Structured trajectories are the natural supervision unit: the planner learns from success/failure verdicts, the verifier learns from state deltas, the low-level policy is retrained on bounded rollouts with cleaner reward signals.

## Why it matters

- **Recovery becomes first-class.** Failure modes get *logged* rather than swallowed by an end-to-end policy that doesn't know it failed.
- **Composition across embodiments.** Because the low-level VLA is behind the interface, the planning half generalizes across robots.
- **Evaluation-ready.** The interface's four event types are natural axes for benchmark decomposition — memory-dependent tasks (RMBench: 12.5% average) versus in-scope manipulation (RoboTwin 2.0: 86.20%, LIBERO: 97.40%) get separable numbers.

## Gotchas & tricks

- **Precondition / postcondition specs are the bottleneck.** They are effectively per-skill mini-verifiers; writing them by hand does not scale. Practical systems learn or generate them alongside the skill library.
- **Bounded execution needs a sensible bound.** Too tight and legitimate skills abort; too loose and the interface's guarantees weaken.
- **The verification step wants a different observation than execution.** Post-hoc verifiers benefit from a wide-angle or third-person view the low-level VLA never sees during action generation.

## Sources

- Paper: *EmbodiedSkills: A Unified Framework for Orchestrating, Training, and Deploying VLA Agents* — Zhang et al., 2026 — [arXiv:2609.01281](https://arxiv.org/abs/2609.01281).
