# Agentic RAG

*Taxonomy — retrieval-augmented generation where an LLM agent orchestrates multi-step retrieval, reasoning, and tool use.*

**TL;DR:** Static RAG runs one retrieve→generate pass. **Agentic RAG** turns retrieval into a loop the model controls: it plans, retrieves, verifies, and iterates. The design axis is *what state the loop makes explicit* — implicit chat history (fragile), externalized state objects (SearchOS-style SOCM), or an RL-shaped policy over multi-granularity retrieval actions (GRASP). Modern systems combine both.

**Related taxonomies:** —
**Depth files covered here:** [search-oriented-context](search-oriented-context.md) · [granularity-aware-search](granularity-aware-search.md)

---

## The problem

Single-shot RAG breaks on multi-hop questions, ambiguous queries, and any task where the right answer requires *chaining* several retrievals. Simply looping "retrieve → answer → maybe retrieve again" hits three recurring failure modes:

- **Loop forever on bad queries.** The agent re-issues near-identical searches, wasting budget.
- **Lose track of coverage.** Long chat histories bury what has and hasn't been searched.
- **Ungrounded answers.** Without discipline the final answer drops citations.

Agentic RAG is the class of techniques that keep the loop from degenerating.

## The shared pattern

Every agentic RAG method exposes a **decision loop over retrieval tools** and adds *some* mechanism to stop the loop from wandering. The mechanism can be:

- Better *scaffolding* — externalize the state so the agent can see what's been tried.
- Better *policy* — train the retrieval decisions with RL so the agent learns granularity control and early stopping.
- Better *tooling* — richer action space (semantic + lexical + reading), or a middleware harness that catches stalls.

The three levers stack.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [search-oriented-context](search-oriented-context.md) | Externalize agent state into Frontier / Evidence Graph / Coverage Map / Failure Memory | Requires designing typed state schemas per task class | Long-horizon open-domain search with many sub-agents |
| [granularity-aware-search](granularity-aware-search.md) | RL over three retrieval tools (semantic / keyword / paragraph) with a composite reward | Needs a dual-index corpus and matched training/deploy budgets | Multi-hop QA where zoom level matters |
| Prompt-only ReAct-style loops (no depth file yet) | Chat-history-native retrieval-generation loop | Fragile past ~10 turns; loop-forever risk | Short, low-stakes queries |
| Self-Reflective RAG (no depth file yet) | Add a verify step that decides whether retrieved evidence is sufficient | Extra latency per turn | Precision-sensitive answers |

## How to choose

Start with prompt-only ReAct for prototypes. When users start hitting loop-forever or coverage-loss failures, add **SOCM-style state externalization** — it's the highest-leverage upgrade. If the corpus admits multiple retrieval modalities, layer **granularity-aware search** on top: RL-trained tool routing is complementary to externalized state, not an alternative.

For frontier deployments, the emerging default is **SOCM + granularity-aware policy + middleware harness**. All three levers pull in different directions and the combination is what you want.

## Adjacent but distinct

- **Static RAG** — one-shot retrieve+generate; not an agent loop.
- **Long-context reading** — cram retrieved docs into the context and let the model attend; no explicit retrieval decisions.
- **Tool-using agents (general)** — agentic RAG is the retrieval-specific slice of tool-using; agent frameworks broadly cover code exec, browser use, etc.

## Sources

- Paper: *SearchOS-V1* — Renmin U / Ant Group, 2026 — externalized state pattern.
- Paper: *GRASP* — UMass Amherst / Adobe Research, 2026 — RL over granularity-aware retrieval actions.
