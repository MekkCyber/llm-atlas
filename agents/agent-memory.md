# Agent Memory with Explicit State (EvoMemState)
*Depth — persist facts with explicit validity, scope, and relational-link state instead of flat vector-DB retrieval.*

**TL;DR:** Default personal-agent memory dumps facts into a vector DB and retrieves by similarity, which loses validity ("did this fact expire?"), scope ("does this fact apply here?"), and relations ("does this fact depend on another?"). **EvoMemState** (Xiang et al., 2026) attaches each fact to a **Relational State** (validity window, scope, links to other facts) and a **Reasoning State** (how the fact applies to the current query), both updated by RL. Reports gains on long-term dialogue and memory-use benchmarks, with the largest gains on retrieval *completeness* rather than raw accuracy — the states let the retriever return the right *subset*, not just a semantically close one.

**Prereqs:** [agents/README](./README.md), [rlvr](../post-training/rlvr.md)
**Related:** [harness-training](./harness-training.md), [latent-context-compression](./latent-context-compression.md)

---

## What it is

Three failure modes of flat-memory personal agents motivate the design:

- **Stale facts** are retrieved as if current ("the user's phone number is X" from two years ago).
- **Out-of-scope facts** are retrieved on topically-similar but contextually-wrong queries ("the user prefers tea over coffee" in a work meeting about tea exports).
- **Relational blindness** means fact A can be retrieved while its invalidating fact B is left behind.

EvoMemState attaches explicit *state* to each memory entry, maintained by an RL loop that updates the state as new facts arrive.

## How it works

### The two state types

**Relational State** (per fact):

- `valid_from` / `valid_until` timestamps or logical conditions.
- `scope`: set of topic / context tags.
- `links`: pointers to other facts (supersedes, depends-on, contradicts, extends).

**Reasoning State** (per query, computed at retrieval time):

- Which of the retrieved facts is *currently valid*, *in scope*, and *not superseded*.
- A ranked subset handed to the agent's context.

### Three components

1. **Ontology-guided state induction.** When a new fact arrives, an LLM-driven inducer decides which state fields to populate, using a fixed ontology (person-attribute, event, preference, task). The ontology is small and hand-designed.
2. **RL-based state evolution.** As new facts arrive, their arrival triggers updates on existing facts' states (invalidate, scope, link). The update policy is trained with RL against a task reward — if the agent's downstream answer improves when fact A was invalidated at the right time, the update policy gets credit.
3. **State-aware retrieval + reasoning.** At query time, filter candidates by the Reasoning State before the LLM sees them. The agent operates on a cleaner context.

## Why it matters

- **Personal-agent memory is a shipping product surface.** ChatGPT memory, Claude Projects, Copilot personalization all run on this problem. Flat vector DBs leak stale and out-of-scope facts into retrieval; explicit state is the first principled fix.
- **Retrieval completeness beats retrieval similarity.** On benchmarks where facts invalidate each other, the EvoMemState-style retriever returns the *right subset* even when a semantically-closer stale fact exists.
- **Debuggable.** Each state field is inspectable, auditable, and editable by hand or policy. Unlike a vector-DB retriever's score, you can explain *why* a fact did or did not surface.

## Gotchas & tricks

- **Ontology design is load-bearing.** A bad ontology cascades through every fact's state. Hand-design the ontology for the target product (personal assistant vs coding agent vs research agent).
- **State induction can lag.** If the LLM-driven inducer is slow, every fact write incurs latency. Batch or asynchronous induction is the fix in production.
- **RL on state evolution is sparse.** The reward signal only lands on queries that use the fact. Mitigation: rollout synthetic queries over newly-added facts to densify the training signal.
- **State-aware retrieval is only as good as the state.** If `valid_until` is wrong, retrieval is wrong. Audit state updates on high-value facts.
- **Not a substitute for vector retrieval.** The state is a *filter* on top of similarity search, not a replacement. Pure state-based retrieval scales poorly in the number of facts.

## Sources

- Paper: *EvoMemState: Evolving Memory State for Personal Agents* — Xiang, Tang, Liu, Chen, et al., 2026 — [arXiv:2611.03743](https://arxiv.org/abs/2611.03743).
- Background: *MemGPT* — Packer et al., 2023 — hierarchical paged memory for LLM agents.
- Background: *A-MEM* — 2024 — vector-DB personal memory baseline.
