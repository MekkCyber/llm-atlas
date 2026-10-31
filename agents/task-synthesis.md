# Agent Task Synthesis (WorkFactory)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A pipeline for turning short *professional intents* into fully executable agent tasks for SFT. The core move is **information-level planning**: build an explicit *information support graph* for each intent — what information the task needs, where it lives, what depends on what — before generating the task's environment. Probe executions then diagnose which generated tasks have broken environments and *repair* them while preserving intent. Yields 112,686 tasks across 612 occupations; SFT on teacher trajectories improves Qwen variants by hundreds of Elo on GDPval-AA v2.1.

**Prereqs:** [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** none

---

## What it is

The limiting resource for agent SFT at scale is not task *descriptions* but *executable* tasks whose environments are correctly provisioned and whose teacher trajectories are reliable. Hand-authoring a few hundred tasks is tractable; auto-generating 100k from short prompts is where the environment generation falls apart.

WorkFactory's framing: an *intent* ("process this travel reimbursement") underspecifies the environment it needs (expense policy document, approval workflow lookup, email templates). The environment is a function of the information support the task requires, which the method makes explicit.

## How it works

### Information-level planning

For each professional intent:

1. **Build an information support graph.** Nodes: facts, documents, tools, email contents. Edges: dependencies ("the reimbursement decision depends on the policy document and the submitted expense").
2. **Instantiate the environment** by walking the graph: synthesize each node (document contents, tool outputs) consistent with its dependencies.
3. **Define the task** against the instantiated environment.

### Probe executions and repair

1. **Probe-run the task.** A probe agent attempts the task against the environment; failures are categorized by *information support defect*: missing node, inconsistent dependency, under-specified goal.
2. **Repair.** Patch the environment to resolve the defect, keeping the intended work unchanged. Not "generate a new task"; the identity of the task is preserved.
3. **Re-probe.** Loop until the probe succeeds or the task is dropped.

### Scaling the output

- **112,686 tasks across 612 occupations.** The library is diverse along the professional-work axis rather than the capability axis.
- **Teacher demonstrations collected on repaired tasks.** Reliability comes from the environment being correctly provisioned, not from retrying teacher rollouts.
- **SFT target: Qwen3 at 4B, 27B, 35B-A3B.** Produces the WorkPilot family.

## Why it matters

- **Agent SFT bottleneck.** The dominant failure mode for scaled agent SFT isn't model capacity; it's bad data. Tasks whose environments are inconsistent produce teacher trajectories that fail, SFT on which teaches confused behavior. WorkFactory addresses the data-generation step directly.
- **Numbers:** WorkPilot-35B-A3B rises from **879 → 1565 Elo on GDPval-AA v2.1**; all three sizes improve over their respective base Qwen3 models across four professional-work benchmarks.
- **Modular with other recipes.** The task-synthesis stage is upstream of trajectory collection and SFT — any rejection-sampling or RL loop can consume WorkFactory tasks.

## Gotchas & tricks

- **Information support graph depth matters.** Shallow graphs produce environments that look complete but are internally inconsistent. The paper's probe-repair loop catches many but not all of these.
- **Occupation coverage ≠ capability coverage.** 612 occupations is lateral diversity (different domains); vertical diversity (same domain, harder tasks) still has to come from capability curriculum.
- **Diminishing returns at larger models.** Component analyses show model-dependent scaling and execution length that first increases then decreases.
- **Repair must preserve intent.** The spec is a one-sentence intent, so "repair the environment, keep the task" is only well-defined if the probe diagnosis is specific. Vague failures lead to task drift.
- **GDPval-AA v2.1 is the primary eval**; downstream reports should check that other agent benchmarks move similarly.

## Sources

- Paper: *WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training* — T. Chen, J. Deng, Q. Wang et al. (IQuest Research + Zhejiang / Renmin GSAI / SIAT / SJTU / PKU), 2026 — arXiv:2610.37161.
- Code / data / models: https://github.com/IQuestLab/WorkFactory
