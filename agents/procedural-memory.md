# Procedural Memory for LLM Agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Procedural memory is a growing store of learned *skills* or *workflows* that an LLM agent accumulates and reuses across tasks. Each skill is a compact description of *how to do* a recurring pattern (a fill-out-this-form procedure, a database-query template, an incident-triage script), grounded in past execution traces. The agent retrieves relevant skills at planning time and refines them after each run. AFTER (Belikova et al., 2026) is the first controlled benchmark for procedural-memory transfer across tasks, roles, and model backbones.

**Prereqs:** [README](README.md)
**Related:** [../evaluation/README](../evaluation/README.md) · [../post-training/rejection-sampling](../post-training/rejection-sampling.md)

---

## What it is

Modern LLM agents (Cursor's Composer, Devin, MetaGPT, enterprise workflow agents) all bolt on some form of *memory*: episodic memory (past conversations), semantic memory (facts), and *procedural memory* — reusable skills for how to do recurring things. Procedural memory is the least-studied of the three but the most operationally useful in enterprise settings, where the same class of task (create ticket → assign → follow up) recurs constantly with small variations.

A **skill** is typically a structured artifact:

```
skill:
  name: "cross-check invoice against PO"
  trigger: "invoice + PO available in context"
  steps: [ fetch PO details, extract line items, compare totals, flag mismatches ]
  known-failures: [ multi-currency lines, PO amendments ]
```

The agent retrieves skills at planning time (usually by embedding similarity or LLM-based routing) and executes them; after execution the trace is used to refine the skill.

## How it works

### Skill lifecycle

1. **Bootstrapping.** Distill an initial skill from a small set of successful traces (LLM prompt: "summarize the recurring pattern in these executions as a reusable procedure").
2. **Retrieval.** At planning time, retrieve top-k skills by embedding similarity to the current task.
3. **Application.** Include the retrieved skill(s) in the planner prompt; the agent adapts them to the current task.
4. **Refinement.** After each execution, evaluate the trace. If the skill was applied and succeeded, keep it as-is; if it partially failed, generate a refined version.

### The AFTER benchmark protocol

AFTER provides **382 realistic enterprise tasks** across **6 professional roles** and **22 procedural skills**, with four controlled evaluation settings:

- **Local improvement:** same task, refine skill in-role.
- **Cross-task transfer:** different task, same role.
- **Cross-role transfer:** same skill class, different role.
- **Cross-model generalization:** skill learned on one backbone, applied on another.

A single refinement round gives **+3.7 to +6.7 aggregate points**. Skills evolved from **diverse multi-model traces** reach **73.1% cross-model test accuracy**, beating skills evolved from single-model traces.

## Why it matters

- **Enterprise agents are recurring-task-shaped.** Enterprise workflows are 80% recurring patterns; procedural memory is exactly the abstraction that lets an agent get better on the tenth invoice without human labeling.
- **Cheaper than fine-tuning.** No weight updates, no RL, no per-role fine-tune. The agent's static weights stay put; procedural memory does the adaptation.
- **Cross-model transferable.** AFTER's headline result is that skills built from *heterogeneous* execution traces (mixed model backbones) generalize better than skills built from a single backbone — a design pointer for skill-library construction pipelines.

## Gotchas & tricks

- **Some skills specialize; some transfer.** AFTER finds that some skills generalize broadly across roles and models while others become role-specific and lose transfer. Track cross-role performance during refinement; don't over-fit to one role's phrasing.
- **Diverse-trace refinement wins.** Skills evolved from traces produced by multiple different models transfer across models much better than single-source skills. If you can afford it, run the refinement over rollouts from a mix of backbones.
- **Retrieval quality dominates.** Bad skill retrieval sends the wrong template into the planner; the agent can degrade below its zero-shot baseline. Instrument retrieval-precision as a first-class metric.
- **Prompt bloat.** A skill library grows quickly. Cap the number of skills injected into any prompt; rely on retrieval instead of stuffing.

## Sources

- Paper: *Managing Procedural Memory in LLM Agents: Control, Adaptation, and Evaluation* — Belikova, Parchiev, Egorov et al., 2026 — the AFTER benchmark and cross-model / cross-role results.
- Related: Voyager, Reflexion, Cursor Composer memory documents (industry recipes for skill libraries).
