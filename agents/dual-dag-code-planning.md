# Dual-DAG Code Planning (Repo0)
*Depth — maintaining explicit architectural state as coupled requirement/component DAGs for zero-to-all repository code generation.*

**TL;DR:** Most repo-level code-generation agents assume a predefined project structure. Real "zero-to-all" generation — build the entire repository from a natural-language spec — needs to *co-evolve* structure and code. **Repo0** maintains an explicit architectural state as a **Dual-DAG**: a requirement-level DAG, a component-level DAG, and an alignment relation between them. The agent takes **structural actions** (split, save, revise, merge, add) driven by modularity metrics until structural convergence, then runs test-driven development to fill code. Vs. the strongest planning baseline (RPG) on six RepoCraft repos with GPT-5 mini and DeepSeek V3.2: **+20.08 pp** Functionality Coverage, **+29.74 pp** Pass Rate.

**Prereqs:** none in the graph yet; assumes standard planner-executor agent basics.
**Related:** [terminal-task-synthesis.md](terminal-task-synthesis.md), [../evaluation/humaneval.md](../evaluation/humaneval.md), [../evaluation/livecodebench.md](../evaluation/livecodebench.md)

---

## What it is

A code-generation agent whose *state* isn't just a set of files but an explicit architecture:

- **Requirement DAG.** Nodes are functional requirements derived from the natural-language spec; edges capture dependency.
- **Component DAG.** Nodes are code components (modules, classes, files); edges are code-level dependencies.
- **Alignment relation.** Maps requirement nodes to the component(s) that implement them, and vice versa.

The Dual-DAG is a first-class artifact the agent reads, mutates, and reasons over — separate from and prior to any code file.

## How it works

Two phases:

**Phase 1 — Structural evolution (until convergence).** The agent repeatedly picks a structural action to apply based on modularity metrics computed over the current Dual-DAG:

- **split** — decompose an over-loaded component into two coupled by an interface.
- **save** — accept the current node.
- **revise** — rename / retype a component to sharpen its role.
- **merge** — collapse redundant components.
- **add** — introduce a missing component required by an unfulfilled requirement.

Iteration continues until modularity metrics stop improving.

**Phase 2 — Test-driven code generation.** The converged component DAG is the scaffold. For each component, the agent generates tests from the aligned requirements, then generates code to pass them.

## Why it matters

Repo-level agent evaluation (see SWE-bench Science and others) is exposing that success requires *architectural* competence, not just per-function correctness. Repo0's Dual-DAG is a specific answer to "what should the agent's persistent state look like above the file system?" — likely a template for general planner-executor agents where the plan itself must evolve.

## Gotchas & tricks

- Modularity metrics drive Phase 1; wrong or gameable metrics produce architecturally-clean but functionally-empty scaffolds.
- The alignment relation is where drift accumulates — as components split/merge, requirement→component links must be updated atomically; stale links poison Phase 2 tests.
- Convergence isn't guaranteed on ill-specified requirements; the paper's real-world repos come with structured RepoCraft specs.

## Sources

- Paper: *Repo0: Design-Driven Zero-to-All Code Generation* — Chen, Teng, Gu et al., 2026 — [arXiv:2608.19854](https://arxiv.org/abs/2608.19854)
