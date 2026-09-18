# Agora — Git as Shared Memory for Multi-Agent Research
*Depth — a coordination substrate for autonomous research swarms in which every claim is a Git commit.*

**TL;DR:** Multi-agent research systems fail at coordination — N independent agents mostly redo each other's work. Agora reuses **Git** as the shared memory: every result, insight, hypothesis, verification, and report is an **immutable commit** whose parent edges name what it builds on. A derived index exposes the frontier, neglected branches, and per-claim verification status; a diversity-aware selection rule prevents mode collapse. Demonstrated by a 12-day run of 13 LM workers with no assigned tasks and no central planner.

**Prereqs:** [README](README.md)
**Related:** [../systems/README.md](../systems/README.md) · [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

A shared, append-only DAG for autonomous research. Nodes are commits carrying one atomic scientific artifact each — a hypothesis, a checked reproduction, a measured result, a written report. Edges are Git's own parent pointers, which encode "this claim builds on those claims." Reproducibility follows for free: any commit can be checked out and rerun.

Agents are decoupled from each other; the graph is the communication.

## How it works

- **Commit types.** Each contribution class (result / insight / hypothesis / verification / report) is a commit template. Payload is structured so machines can index it.
- **Index.** A derived read model of the graph exposes:
  - the **frontier** (leaf commits open for extension),
  - **neglected branches** (low-fanout subtrees for diversification),
  - **verification status** per claim (how many independent reproductions).
- **Selection rule.** Workers choose what to work on next by a **diversity-aware** policy — probability of extending a subtree decays with its current fanout, so the community doesn't collapse onto one leader.
- **Reproduction is a commit.** A worker rerunning another's claim commits a verification node; independent reproductions accumulate.

## Why it matters

- **First substrate where marginal agents ≈ marginal discoveries.** In the 12-day, 13-worker run, agents published **1,703 contributions**, drove eval from 3.39 → **1.899 bits per byte** on a weight-transfer task (init a frozen 119.6M attention-SSM hybrid from 141 donor models, no training data, no gradients), closing **62%** of the gap to a trained GPT-2 124M.
- **Every claim is auditable.** 145-commit winning ancestry, 15 contributor accounts, **165 independent reproductions, none failed.**
- **Coordination without a planner.** No central scheduler, no assigned tasks, no reward broadcast — the graph plus the selection rule is enough.

## Gotchas & tricks

- **Diversity rule is load-bearing.** The paper reports one mid-run human intervention pulling the swarm out of a monoculture. The selection policy is not a set-and-forget primitive.
- **Verification is voluntary.** The graph rewards new claims and new verifications equally on paper, but agents can chase novelty and under-invest in reproduction. Tuning the reward is future work.
- **Merge conflicts vanish.** Since every commit is atomic and additive, there are no code-level merges — the DAG stays clean by construction.
- **Doesn't establish causality.** The paper flags this: shared memory correlated with discovery, but the controlled comparison against N independent single-agents is not run.

## Sources

- Paper: *Agora: Git as Shared Memory for Collective AutoResearch* — Zhang, Zou, Zhang, Hu, Zhang, Xu, Kautz, Dong, NVIDIA + collab, 2026, arXiv 2609.18094.
