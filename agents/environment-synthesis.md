# Tool-Centric Environment Synthesis (ToolSpace)
*Depth — procedurally generate verifiable tool-use training tasks by mining dependency structure from MCP service traces.*

**TL;DR:** Agent-RL is data-starved on long-horizon tool compositions. ToolSpace inverts the usual flow: instead of starting from a database and defining tools over it, it **mines data / state / retry dependencies from real MCP service call traces** using deterministic rules, clusters the resulting dependency graph into coherent environments, and generates verifiable multi-step tasks via **state-aware random walks** through the graph. Produces harder "Pro" versions of existing benchmarks and lifts agent performance on seven tool-use benchmarks, with scale curves that improve with environment diversity.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [README.md](README.md) · [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

Training long-horizon tool-use agents requires environments (tools + state + verifier) that compose over many steps. Hand-authored environments don't scale: each new tool-and-use-case costs human effort, and verifiers don't generalize. The result is a thin long-tail of compositions in training sets and a wide gap between training and real deployment.

ToolSpace reframes environment construction as **trace mining**: observable MCP (Model Context Protocol) service traces in the wild contain the latent dependency structure of tool compositions. Reverse-engineering that structure gives both the environment (tool graph + state transitions) and the verifier (terminal state check) for free.

---

## How it works

Three-stage pipeline:

1. **Dependency mining from traces.** Collect MCP service call traces (real agent-tool interactions, no annotations). Apply deterministic rules to extract three dependency kinds per tool pair:
   - **Data dependency** — tool B consumes the output of tool A.
   - **State dependency** — tool B requires a state created by tool A.
   - **Retry dependency** — tool B is called only when tool A fails.

   The output is a labeled directed graph over the tool set.

2. **Environment clustering.** Cluster connected subgraphs into coherent *environments* — a bounded set of tools that compose naturally (e.g., an email+calendar environment). Each cluster becomes one training environment.

3. **Task generation via state-aware random walks.** Inside an environment, sample a random walk through the dependency graph, starting from an initial state and ending at a goal state. The sequence of tool calls defines the task; the terminal state check is the verifier (reachable from the random walk itself). Call this a **verifiable multi-step tool-use task**.

Training is standard RLVR with the generated verifier — no learned critic, no human labels.

---

## Why it matters

- **Scales with trace volume.** The bottleneck is MCP trace coverage, not human task-writing. Where traces exist, environments appear.
- **Verifier-free-of-annotation.** The terminal state check comes from the generation process. No reward hacking surface via a learned critic.
- **Composition diversity via graph sampling.** Random walks naturally cover long tails of compositions that hand-curation misses.
- **Reproducible "Pro" benchmark variants.** The paper applies the pipeline to existing environments (AutomationBench, τ³-Bench Banking) and generates harder versions whose difficulty is a function of walk length, not human judgment.
- **Agent-RL training set scale is the gating resource for the next generation of tool agents.** ToolSpace is a template for how to produce that scale without a human loop.

---

## Gotchas & tricks

- **Trace quality dominates.** Noisy traces (failed calls, tools miscalled) pollute the dependency extraction. Filter aggressively before mining.
- **Dependency-rule precision over recall.** The deterministic rules are tuned for high precision (correct dependencies, miss some) rather than high recall. Missing some dependencies is cheap; inventing wrong ones generates unsolvable tasks.
- **Random-walk length is the difficulty knob.** Short walks give easy tasks and saturate quickly; too-long walks become undirected. The paper reports a bell-curve of difficulty vs training signal around 6–12 step walks.
- **Clusters leak when tools are overloaded.** A tool that appears in many clusters (e.g. generic HTTP fetch) makes clustering unstable. Mark such tools as "connector" and treat specially in random-walk edges.
- **State verifier must be deterministic.** Non-deterministic terminal states (e.g. "any correct email was sent") need an explicit canonicalization step before checking.
- **Not a replacement for real-world eval.** Synthetic walks capture *compositions*, not real-user intents. Pair with human-authored eval sets at release.

---

## Sources

- Paper: *ToolSpace: Scaling Tool-Centric Environment Synthesis for Long-Horizon Agents* — Sun, Yang, Wang et al. (Renmin, Fudan, SJTU), 2026, arXiv 2611.05036 — primary source.
- Related: [../post-training/rlvr.md](../post-training/rlvr.md) — verifiable rewards from executable state checks; ToolSpace produces the verifiers used in RLVR for tool-use.
