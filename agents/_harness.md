# Agent Harness

*Taxonomy — the structured execution layer around a foundation model that turns single-turn model capability into long-horizon agent behavior.*

**TL;DR:** A modern LLM agent is not just "a model with tool calls." It is a *harness*: an architecture of substrates (context constructor, memory, skill/tool router, orchestration loop, verification & governance) that surrounds the foundation model. Agent quality emerges from the *interaction* between these substrates, not from any single one. Evaluating only final-task accuracy hides where harnesses break — this taxonomy names the layers so they can be designed, measured, and improved as first-class objects.

**Related taxonomies:** [_attacks](../safety/_attacks.md) · [_scheming](../safety/_scheming.md)
**Depth files covered here:** [direct-corpus-interaction](direct-corpus-interaction.md) · *(more depth files to come)*

---

## The problem

Frontier-model gains on single-turn benchmarks have decoupled from agent performance on long-horizon tasks. Long-horizon agents fail in ways the model can't fix — memory poisoning, context bloat, tool-routing thrash, untrusted-content escape, orchestration deadlock. These are *system* failures, not model failures. Yet most evaluation collapses agents into a single end-to-end success metric, which gives no signal on which substrate is the bottleneck.

The harness frame says: treat the layers around the model the way we treat the model itself — with explicit interfaces, ablations, and benchmarks. Otherwise the field optimizes a black box and stalls.

---

## The shared pattern

Every modern agent harness has roughly five layers:

```
┌──────────────────────────────────────────────────────────┐
│ Verification & Governance     (audit, policy, rollback)  │
├──────────────────────────────────────────────────────────┤
│ Orchestration loop            (planning, retries, halt)  │
├──────────────────────────────────────────────────────────┤
│ Skill / tool router           (which tool / sub-agent)   │
├──────────────────────────────────────────────────────────┤
│ Context constructor           (prompt assembly, RAG)     │
├──────────────────────────────────────────────────────────┤
│ Memory substrate              (read/write, provenance)   │
└──────────────────────────────────────────────────────────┘
                  ↕ foundation model ↕
```

The foundation model is consulted by every layer; the layers do not reduce to "extra prompts." Each has its own state, lifecycle, failure modes, and (eventually) its own metric.

---

## Variants (bottlenecks)

| Layer | Key concern | Failure mode | Open eval direction |
| --- | --- | --- | --- |
| Memory substrate | Provenance, trust, decay | Poisoned writes survive sessions | Memory hygiene tests, write-audit traces |
| Context constructor | Token budget, relevance, salience | Context bloat / lost-in-the-middle | Context efficiency, signal density |
| Skill / tool router | Coverage, latency, mis-routing cost | Tool thrash, hallucinated tools | Routing precision/recall, tail latency |
| Orchestration loop | Step budget, halting, error handling | Infinite loops, premature halt | Trajectory quality, step-economy |
| Verification & governance | Safe action, policy compliance, audit | Untrusted-content escape, persistent-control trojan | Action allow-list adherence, provenance tracking |

Each row is a candidate concept area for the graph; depth files will accumulate as papers carve them out.

---

## How to choose (as a designer)

Ship the simplest harness that fits the task:

- **Single-turn QA with a retriever** — context constructor + a thin orchestration loop. No persistent memory.
- **Multi-step research / tool agent** — add a skill router (more than ~5 tools) and a step-budget controller.
- **Stateful workspace agent (code, IDE, OS-level)** — full stack, with hardened verification & governance because workspace state is sticky. See [persistent-control-attack](../safety/persistent-control-attack.md).
- **Multi-agent population** — add inter-agent communication monitoring; see [emergent-language-evasion](../safety/emergent-language-evasion.md).

A common mistake is to add memory and tool-routing before the orchestration loop is solid — the upper layers can't compensate for an unbounded or non-halting controller.

---

## Adjacent but distinct

- **Foundation-model post-training** ([_post-training](../post-training/_post-training.md)) makes the model better at *each call*. Harness design makes the *system* better at composing calls. The two are complementary.
- **RL-on-agents** (training the policy on long-horizon trajectories) treats the harness as the *environment*. The harness sets the action space — its design upper-bounds what the policy can ever learn.
- **Safety / monitoring** ([_attacks](../safety/_attacks.md)) intersects with the verification-and-governance layer but is broader: it includes attacks on the model itself, not just on the harness.

---

## Sources

- Paper: *From Model Scaling to System Scaling: Scaling the Harness in Agentic AI* — Shangding Gu, UC Berkeley, 2026 — coins "harness" as a first-class object and proposes the layer decomposition above.
