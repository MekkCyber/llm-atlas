# Mental World Modeling (MWM / Mentis)
*Depth — world models that track mental state (belief, desire, intent) as first-class variables.*

**TL;DR:** Standard world models answer physical questions — what/where things are, how they evolve. Human behavior is driven by *mental* state (what agents believe, want, intend, feel). A world model that gets the scene right but the mental state wrong predicts the wrong action for the right-looking scene. **Mental World Modeling** couples a physical world state with a mental world state, renders target-specific partial observations, and simulates how actions update *both* components. **Mentis** is a training-free, inspectable baseline that outperforms 8 LLM-based world models on a curated situated-decision dataset.

**Prereqs:** *(none — foundational)*
**Related:** [pt-flow.md](../multimodal/pt-flow.md)

---

## What it is

A formulation of world modeling that treats mental variables as core state, not post-hoc rationalization. The world state is `(physical, mental)`; observations are target-specific projections of that joint state; transitions update both under a candidate action. This differs from "theory of mind evaluation" work — MWM is about the *world model itself*, not about probing whether an LLM can guess someone's belief.

## How it works

**State.** `s = (s_phys, s_ment)` — physical scene (objects, positions, dynamics) and mental state per-agent (belief, desire, intent, felt-emotion, social permissibility).

**Observation.** `o = π(s; target)` — a target-specific partial observation. Different agents in the scene see different things; the observation projection encodes that asymmetry.

**Transition.** `s' = T(s, a)` — an action updates *both* physical (via physics) and mental (via belief-update rules and perceived affordances).

**Mentis (training-free instantiation).** Decomposes MWM into inspectable stages that can be individually swapped:

1. **State parsing.** Parse the scenario into `(s_phys, s_ment)`.
2. **Target-observation generation.** Compute what each target agent perceives.
3. **Action decomposition.** Enumerate candidate actions.
4. **Coupled physical-mental transition.** Simulate each action's joint effect.
5. **Branch-level value evaluation.** Score each resulting state and pick.

Each stage is a call to an LLM prompted for that specific subtask, so every intermediate is human-readable.

## Why it matters

- Bridges agent-oriented evaluation and world-model research: agents that reason about other agents' beliefs need a world model that *tracks* those beliefs.
- Inspectability comes for free — separating the stages exposes exactly which mental variables an LLM is failing to track.
- On the curated situated-decision dataset (text, image, sounding-video stories), explicit mental-state modeling is *necessary* — 8 standard LLM world-model baselines underperform Mentis.

## Gotchas & tricks

- The mental-state ontology (belief / desire / intent / felt-emotion / social permissibility) is a design choice; different ontologies give different failure modes.
- Training-free is a feature for inspectability but a bottleneck for scale — future work needs to distill Mentis-style structured inference into a single trained model without losing the stage-level auditability.
- The evaluation dataset is manually curated; larger, less-curated data may re-expose the same failures the baselines had.

## Sources

- Paper: *Mental World Modeling* — Fei, Zhao, 2026 — [arXiv:2607.27201](https://arxiv.org/abs/2607.27201)
