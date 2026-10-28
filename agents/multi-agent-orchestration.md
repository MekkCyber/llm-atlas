# Multi-agent orchestration — train the planner, freeze the specialists
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pattern for multi-agent systems where one "main agent" writes short, free-form executable workflows that call fixed sub-agents and verify their outputs. Only the main agent is trained (SFT then refinement); the sub-agent zoo is modular and swappable without retraining. DynaFlow instantiates this pattern and reports **+10.9 pp** over the strongest baseline across six math-reasoning, scientific-QA and multi-hop-retrieval benchmarks (88.08% overall with a Qwen3.5-27B main agent).

**Prereqs:** [../agents/README.md](../agents/README.md), [../post-training/README.md](../post-training/README.md)
**Related:** [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Multi-agent systems typically fall into two buckets. In **static orchestration** (LangGraph-style) a human writes a DAG of agent calls; the LLM only fills in the nodes. In **LLM-as-router** systems a frontier LLM picks the next sub-agent at every step. Static graphs don't generalize across task shapes; LLM-as-router is expensive at inference and hard to improve with training.

Workflow-synthesis orchestration takes a third route: at the start of each task, the main agent emits a **short executable workflow** — a program whose primitives are calls to sub-agents and verification steps. Execution then runs the program. Only the main agent (the "programmer") is trained.

## How it works

1. **Workflow language.** A compact DSL (or Python-subset) whose primitives are `call(sub_agent, args)` and `verify(output, spec)`. Branching and loops are allowed; the vocabulary of sub-agents is fixed per deployment.
2. **Main-agent SFT.** Collect (task, workflow, outcome) tuples from a strong policy (a frontier model + hand-written workflows), keep the ones where the workflow executed and solved the task, train the main agent to produce such workflows token-by-token.
3. **Refinement.** A second pass that improves the main agent on tasks where the SFT policy's workflow failed — the paper calls this "refinement"; conceptually it is targeted SFT on the harder tail. No sub-agent is updated.
4. **Inference.** Given a task, the main agent emits a workflow; the runtime executes it; verification steps gate sub-agent outputs; the final answer is the workflow's return value.

## Why it matters

Decoupling planning from specialist execution means:

- **New sub-agents cost nothing to add** — they do not require re-training the orchestrator, only exposing them in the DSL.
- **Lower inference cost than LLM-as-router.** One workflow synthesis per task, not one routing decision per step.
- **Cleaner evaluation.** Workflow bugs are a static artifact — you can diff and inspect them.

## Gotchas & tricks

- **DSL design is the hidden ML work.** Too narrow and the main agent can't express complex tasks; too wide and training data is sparse.
- **Verifier steps matter.** Without `verify(...)` inside the workflow, sub-agent errors propagate silently. The paper uses verification steps aggressively.
- **Not for interactive settings.** The pattern assumes the task is specified upfront. Streaming, open-ended agents still want runtime routing.
- **Reported numbers are "Forced mode"** — the main agent must emit a workflow, not fall back to direct answering. "Free mode" numbers aren't the headline.

## Sources

- Paper: *DynaFlow: Learning Free-Form Multi-Agent Orchestration* — Liu, Li, Wang, Wang, National University of Singapore, 2026 — [arXiv:2610.34467](https://arxiv.org/abs/2610.34467).
