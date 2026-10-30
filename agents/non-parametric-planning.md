# Non-Parametric Planning
*Depth — retrieval-grounded LLM planning that adapts without updating model weights.*

**TL;DR:** **Non-parametric planning** stores task-relevant knowledge in an external store (schemas, trajectories, past plans) rather than in the LLM's weights, and *retrieves + grounds* it at inference time to drive planning. The adaptation surface is the store, not the model. CausGround (Shen et al., 2026) is the clearest recent instance: a causal-interaction schema store plus retrieval-grounded prompting lifts embodied-planning success by 10–30 points on VirtualHome and RoboCasa without any gradient update. The pattern generalizes beyond embodied agents to any task where "the model is stable but the knowledge isn't."

**Prereqs:** [causal-action-schema.md](causal-action-schema.md), [computer-use-agents.md](computer-use-agents.md)
**Related:** [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Classical approaches to improving an agent's planning sit on two axes:

| Axis | Where adaptation lives |
| --- | --- |
| **Parametric** (SFT, RL, VLA fine-tune) | model weights |
| **In-context** (few-shot, CoT prompting) | prompt only |
| **Non-parametric** | external store + retrieval |

Non-parametric planning uses an external **knowledge store** — plans, demonstrations, causal action schemas, API docs, procedural manuals — and a retrieval mechanism to inject the relevant subset into the LLM's context at inference time. The LLM provides general reasoning; the store provides task-specific knowledge.

Related to RAG but distinct: RAG retrieves text passages to answer a question; non-parametric planning retrieves *action-structured* knowledge (preconditions, effects, past plan traces) to produce an executable plan.

---

## How it works

### Store shapes

| Store type | Entry shape | Typical source |
| --- | --- | --- |
| Causal action schema | `(action, preconditions, effects, context)` | [causal-action-schema.md](causal-action-schema.md) |
| Trajectory bank | `(goal, (state, action)*)` | past agent rollouts |
| Procedural manual | natural-language SOP | human-authored |
| Memory of past plans | `(task, plan, outcome)` | agent's own history |

### Query → ground → plan

```
query = encode(goal, current_observation)
hits  = store.retrieve(query, k)
plan  = LLM.plan(goal, observation, hits)
```

The grounding step is the LLM aligning retrieved entries to the current observation (e.g., mapping schema object placeholders to actual object instances in view).

### Update loop

```
after execution:
  if outcome was informative:
    store.add(new entry extracted from trajectory)
```

The store grows monotonically; model weights stay frozen.

---

## Why it matters

- **Cheap adaptation.** No GPU training. Adaptation = append to the store. Minutes, not days.
- **Backbone-agnostic.** Works with any closed- or open-weight LLM; swap backbones without re-collecting training data.
- **Inspectable.** Every planning decision is a function of (a) prompt, (b) retrieved entries. Both are human-readable; errors are debuggable without probing weights.
- **Composable with parametric methods.** A VLA fine-tuned on robot teleop can still consume a non-parametric schema at inference. The two aren't exclusive.
- **Right fit when knowledge changes faster than compute budget allows.** Enterprise agents whose procedures change weekly can keep a current store without re-training the backbone.

---

## Gotchas & tricks

- **Retrieval quality is the ceiling.** A perfect LLM with no relevant retrieval produces no better plan than an LLM with no store. Invest in the retriever (dense + symbolic + task-aware) before scaling the store.
- **Grounding is the hardest step.** Aligning stored symbolic entries to the current observation requires perception the LLM may not have. For embodied agents, pair with a grounded VLM.
- **Store rot.** Entries extracted from old trajectories may describe stale environments (API versions, UI layouts). Timestamp and prune.
- **Context window pressure.** Retrieved entries + observation + plan trace quickly saturates even long-context LLMs. Rank aggressively and summarize.
- **Not a cure-all for weak backbones.** A base model that cannot reason over structured context cannot use the store. Non-parametric planning *lifts* a capable backbone; it cannot *create* one.

---

## Sources

- Paper: *CausGround: Causal Interaction Grounding for Adaptive Embodied Planning* — Shen et al., 2026, [arXiv 2610.29702](https://arxiv.org/abs/2610.29702) — the canonical recent demonstration.
- Related: Voyager (Wang et al., 2023) — early example with a skill library.
- Related: Reflexion (Shinn et al., 2023) — self-reflection as a lightweight non-parametric memory.
- Related: [causal-action-schema.md](causal-action-schema.md) — the specific store shape CausGround uses.
