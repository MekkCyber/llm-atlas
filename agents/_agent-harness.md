# Agent Harness

*Taxonomy — the runtime scaffolding around an LLM agent: what the loop looks like, what state it keeps, and what guarantees it enforces.*

**TL;DR:** An **agent harness** is everything that wraps the raw model call — the tool bus, the memory, the state tracker, the retry logic, the transaction manager. Most long-horizon agent failures are harness failures, not model failures. Modern practice is to name the harness explicitly, version it, and treat "harness scaling" (better scaffolding on an unchanged model) as a first-class capability lever alongside weights and prompts.

**Related taxonomies:** [_rl](../post-training/_rl.md) (harness-agnostic training) · [_post-training](../post-training/_post-training.md)
**Depth files covered here:** [harness-scaling](harness-scaling.md) · [black-box-rl-harness](black-box-rl-harness.md) · [agentic-transactions](agentic-transactions.md) · [gui-agents](gui-agents.md) · [model-routing](model-routing.md) · [subtask-workflows](subtask-workflows.md)

---

## The problem

The model is one step; a long-horizon agent is thousands of steps. Somewhere between "raw `generate()` call" and "finish a 6-hour terminal task" you need durable state, checked transitions, retries, rollback, tool schemas, budget accounting, and observability. That "somewhere" is the harness. Two agents on the same model can differ by 10+ points on a benchmark just from a better harness — enough that comparing model checkpoints without disclosing the harness is nearly meaningless.

## The shared pattern

Every harness answers the same four questions:

1. **What is one step?** A single tool call? A ReAct thought-action-observation triple? A phase transition in a state machine?
2. **What survives across steps?** Just the token history? A structured scratchpad? Named state slots with typed writes?
3. **What is enforced vs advisory?** Are illegal transitions rejected, or does the model just get told "please don't"? Are budgets hard caps or soft nudges?
4. **What is recoverable?** After a crash mid-workflow, do you resume from the last checkpoint, replay from scratch, or start over cold?

Concrete harnesses pick a point in that space; the framework name (LangGraph, AutoGPT, StateM, ClawGym) is really a shorthand for those four answers.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| ReAct loop | Free-form thought → action → observation, model chooses everything | Simple; drifts on long horizons | Prototyping, short tasks |
| [harness-scaling](harness-scaling.md) | Durable state, checked transitions, versioned runbooks | Adds engineering surface; huge on long-horizon | Long-horizon coding, terminal tasks |
| [agentic-transactions](agentic-transactions.md) | ACID guarantees over tool-call workflows | Rollback needs tool-level compensation | Real-world side effects (email, code, payments) |
| [gui-agents](gui-agents.md) | Screen + demonstrations as the harness surface | Screen brittleness | Computer-use, office automation |
| [subtask-workflows](subtask-workflows.md) | Demonstrations → reusable subtask library | Needs a live data engine | Repeated task shapes across users |
| [model-routing](model-routing.md) | Prompt-conditional routing across workflows/models | Router quality caps everything | Cost-latency-quality tradeoffs at inference |
| [black-box-rl-harness](black-box-rl-harness.md) | Train against the harness as an opaque environment | No introspection = weaker credit assignment | Training one policy across many harnesses |

## How to choose

Start with a plain ReAct loop and only add machinery when a specific failure mode shows up. Long-horizon coding drops → add [harness-scaling](harness-scaling.md). Real side effects go inconsistent on crash → add [agentic-transactions](agentic-transactions.md). Cost blows up → add [model-routing](model-routing.md). Same task shape recurs across users → build a [subtask-workflows](subtask-workflows.md) library. The mistake is bolting on all of them up front — you get a framework and no shipped agent.

If you're **training** an agent (not just running it), [black-box-rl-harness](black-box-rl-harness.md) is the pattern that lets one policy generalize across harnesses without per-harness engineering.

## Adjacent but distinct

- **Prompting techniques** (CoT, few-shot, self-consistency) live inside a single call — the harness is what wraps many calls.
- **RL post-training** ([_rl](../post-training/_rl.md)) updates the weights; harness scaling updates the runtime. They compose.
- **Retrieval / memory** systems are one *component* a harness may include; they're not themselves harnesses.

## Sources

- Paper: *StateM: Reaching 95.3% Raw Accuracy on Terminal-Bench 2.1 via Harness Scaling* — Qin et al., 2026 — coined "harness scaling."
- Paper: *ClawGym II: Exploring Black-Box RL on Agent Harness* — Song et al., Renmin U., 2026 — harness-agnostic RL.
- Paper: *Agentic Transaction: Towards ACID-Compliant Agent Systems* — Sun, Wang, Li, Tsinghua, 2026.
