# Causal Action Schema
*Depth — precondition/effect schemas over object attributes and relations, used as structured planning knowledge for LLM-based embodied agents.*

**TL;DR:** A **causal action schema** is a symbolic representation of what each action *requires* (preconditions) and what it *changes* (effects), expressed over a factorized **world-state** of object attributes and inter-object relations. CausGround (Shen et al., ECNU / BAAI, 2026) distills this schema from prior interaction trajectories, then uses it at inference time: an LLM planner grounds task-relevant preconditions and effects against the current observation and reasons forward or backward over them to produce a plan — all without updating model weights. Reports VirtualHome success rising **24.9% → 53.3%** and RoboCasa closed-loop success **43.0% → 57.0%**.

**Prereqs:** [computer-use-agents.md](computer-use-agents.md), [../multimodal/vla.md](../multimodal/vla.md)
**Related:** [non-parametric-planning.md](non-parametric-planning.md), [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md)

---

## What it is

Classical planning (PDDL, STRIPS) uses actions of the form:

```
action: pick(x)
  preconditions: holding() = none, reachable(x) = true, on(x, y) for some y
  effects:       holding() = x, on(x, _) = false
```

A **causal action schema** generalizes this to the embodied-LLM setting:

- The world state is a **factorized** representation: `{object: attributes}` + `{(objectA, objectB): relation}`.
- Preconditions and effects are expressed over this representation.
- Entries are **extracted from trajectories**, not written by hand — a model mines (action, before, after) tuples and generalizes.

The schema is a *reusable knowledge base*, not a frozen symbolic planner — the LLM reads it at inference time and reasons over it in natural language.

---

## How it works (CausGround)

### Extraction (offline)

```
for trajectory τ in interaction corpus:
    for (s_t, a_t, s_{t+1}) in τ:
        diff = factorized_diff(s_t, s_{t+1})
        add (a_t, s_t|_{relevant}, diff) to schema store
```

A diff over the factorized state yields a candidate effect, filtered by frequency / consistency across trajectories.

### Grounding (per task, at inference)

```
goal, observation = task_input()
relevant = retrieve_schema_entries(goal, observation)
plan = LLM.reason(goal, observation, relevant,
                  mode="backward_from_goal" or "forward_from_state")
```

The LLM uses the retrieved schema entries as structured context. Backward reasoning: start from the goal and chain preconditions. Forward reasoning: start from the state and chain effects. Both without any gradient update.

### Non-parametric

The LLM's weights never move. All adaptation lives in the schema store, which grows as new trajectories are observed. This makes CausGround compatible with any backbone (open or closed).

---

## Why it matters

- **Decouples knowledge from the model.** Interaction knowledge updates by writing to the schema; model weights stay frozen. Easier to audit, inspect, and version than fine-tuning.
- **Cheap compared to VLA fine-tuning.** No GPU training. Extraction is a one-pass over trajectories; retrieval is a vector-DB or symbolic lookup at inference.
- **Reusable across tasks and embodiments.** The schema is attached to actions and relations, not to a specific robot. Transfer requires only re-grounding against a new observation space.
- **Measurable gains.** VirtualHome 24.9% → 53.3%; RoboCasa 43.0% → 57.0% closed-loop. Also beats baselines on the SWITCH interaction-reasoning benchmark and in real-robot manipulation.

---

## Gotchas & tricks

- **Factorization quality is the ceiling.** If the perception module can't recover attributes and relations reliably, the schema cannot ground. CausGround uses an off-the-shelf grounded VLM for perception; perception error is the dominant residual failure mode.
- **Spurious schema entries.** A single observed effect may be coincidental ("after pick(cup), light goes off"). Frequency and context filters are needed; CausGround applies both.
- **Forward vs backward reasoning tradeoff.** Backward works for goal-conditioned tasks with reachable preconditions; forward works when the state space is small. Mixing requires a controller — the paper uses the LLM itself.
- **Not a replacement for a planner on long horizons.** Over 20+-step plans, cumulative perception error compounds. Pair with periodic re-grounding or a short-horizon MPC-style wrapper.

---

## Sources

- Paper: *CausGround: Causal Interaction Grounding for Adaptive Embodied Planning* — Shen et al., ECNU / BAAI / NWPU / UCAS / CASIA, 2026, [arXiv 2610.29702](https://arxiv.org/abs/2610.29702) — introduces the schema + grounding framework.
- Related: PDDL / STRIPS — the symbolic-planning ancestry.
- Related: [non-parametric-planning.md](non-parametric-planning.md) — the broader pattern CausGround sits in.
- Related: [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md) — contrast with search-based planning.
