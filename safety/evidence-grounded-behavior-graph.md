# Evidence-Grounded Behavior Graph (EBG)
*Depth — a training-free data structure for monitoring long-horizon agents by grouping source-linked evidence into behaviours and organising their relationships.*

**TL;DR:** Long-horizon agents produce more decisions than any reviewer can read. **EBG** groups source-linked evidence (tool calls, file edits, environment responses) into *behaviours* and organises their relationships into a graph. **Task-oriented views** of this graph are what a monitor LLM actually sees — so monitors spot consequential autonomous decisions and locate the evidence a user would need to verify them, without being drowned in raw trace. Introduced alongside **AgentMonBench** (2026); improves consequential-decision identification and evidence localisation across eight models vs. raw-context monitoring.

**Prereqs:** [cot-monitoring.md](cot-monitoring.md), [README.md](README.md)
**Related:** [_scheming.md](_scheming.md) · [safety-case.md](safety-case.md) · [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

A graph structure for post-hoc or online oversight of long-horizon agents (software-engineering agents, computer-use agents). Three ideas composed:

1. **Behaviours as nodes.** Not raw steps — grouped source-linked evidence (several tool calls plus their outputs that together achieve something: "edited the config", "ran the test suite", "modified an environment variable").
2. **Relationships as edges.** Precedence, data dependency, undo/supersede. The graph makes it visible which behaviours affect which later behaviours.
3. **Task-oriented views.** Monitors never see the whole graph; they see projections that keep only the behaviours relevant to a *consequential-decision* question or *requirements-vs-behaviour alignment* question.

Training-free: EBG is built from attribution already present in trace evidence (tool-call outputs, diff lines, log messages), not from learned embeddings.

## How it works

- **Evidence extraction.** For each agent step, parse the trace into (action, output, source-link) tuples. Source-link is a pointer back to the exact artefact (file, env variable, HTTP response) the behaviour touched.
- **Clustering into behaviours.** Group adjacent evidence by source overlap and semantic similarity, forming behaviour nodes.
- **Graph assembly.** Edges from behaviour $A$ to $B$ when $B$'s evidence depends on $A$'s source artefacts, with explicit relation types (precedence, data-flow, revision).
- **View generation.** For a monitor question $Q$ (e.g. "is this decision consequential?"), filter the graph to the subgraph touching $Q$'s referenced artefacts, with an explicit "others not shown" summary preserving auditability.

A monitor LLM reads the view and emits (a) a consequentiality verdict and (b) evidence pointers back to the full graph.

## Why it matters

- **Oversight as a benchmark target.** AgentMonBench scores monitors on consequential-decision ID and evidence localisation across software-engineering agent traces — the first serious oversight-side benchmark.
- **Views beat raw context.** EBG's gains over raw-context monitoring persist across input scales and hyperparameters; the data structure, not just the model, matters.
- **Audit trail.** The graph is a durable artefact: a user can drill from a monitor's claim back to source-linked evidence, which is exactly what an oversight UI needs.

Expected to leak into agent-safety tooling: the EBG template fits any trace where tool calls and their outputs can be source-attributed.

## Gotchas & tricks

- Behaviour clustering is the ceiling — bad clustering produces either noisy nodes (too granular) or opaque ones (too coarse). Tune on labelled traces per domain.
- Views can hide: a well-chosen projection can also *conceal* a consequential decision the monitor would otherwise catch. Keep a "nothing to see here" audit layer.
- Edge semantics ("undo", "supersede") must be *robust* under agent re-ordering; weak extraction here causes false precedence edges.
- Not a substitute for runtime guardrails — EBG is observational. Pair with action-level moderation when stakes are high.

## Sources

- Paper: *What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents* — Sun, Yan, Zhao, Ding, Zhang, Yang, Zhang, Xu — Renmin University of China (RUC-GSAI-IIRLab), 2026 — [arXiv:2610.06406](https://arxiv.org/abs/2610.06406).
