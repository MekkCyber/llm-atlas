# Typed Completion Graphs (Interlock)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A structured task representation for long-horizon LLM agents that makes *task completeness* a dependency-graph problem instead of a flat predicate. Each task is a typed graph with three edge classes: **grounding** (evidence → decisions), **coupling** (joint constraints between decisions), **realization** (decisions → required effects). The graph compiles to a resettable workspace, a public request, and a private rubric. Fine-tuning on self-generated accepted trajectories (filtered by an agent-as-judge checking all mandatory requirements) improves pass rates +3–7 points across AutomationBench, JobBench, and Harvey LAB.

**Prereqs:** [rlvr](../post-training/rlvr.md), [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [_rewards](../post-training/_rewards.md)

---

## What it is

Agent benchmarks typically score tasks on a flat pass/fail: either every requirement is satisfied or the task fails. For long-horizon work this hides a dominant failure mode — the agent satisfies individual requirements *in isolation* but violates the latent dependencies between them. Decisions aren't grounded in the evidence they claim to use; coupled decisions contradict each other; realised effects don't match stated decisions.

Interlock encodes those dependencies structurally. Instead of writing a rubric as a flat checklist, the task author describes a *completion graph* whose nodes are evidence items, decisions, and required effects, and whose typed edges state the relationships that must hold. A trajectory "completes" the task iff every mandatory edge is satisfied by its artefacts.

## How it works

**Three edge types.** Each edge has a specific verifier:

- **Grounding** (evidence → decision). The decision's artefact must reference the evidence item (quote, citation, structured link). Verifier is cheap: textual match or structured pointer.
- **Coupling** (decision ↔ decision). Jointly-constrained decisions must satisfy an invariant (e.g. "the quoted price in the summary equals the invoice total"). Verifier is a specified predicate.
- **Realization** (decision → effect). The claimed decision must map to an observable effect in the environment (file written, API call made with matching payload). Verifier executes the effect check against the workspace.

**Compilation.** The graph compiles to:
- A **resettable workspace** — the environment the agent operates in, re-initialisable per trajectory.
- A **public request** — natural-language task description generated from the node labels; this is what the agent reads.
- A **private rubric** — the typed edges, invisible to the agent, consumed by the verifier.

**Trajectory filter.** The agent produces a trajectory; the verifier walks the completion graph; any missed mandatory edge rejects the trajectory. Accepted trajectories become SFT data; the policy is fine-tuned on its own accepted trajectories (classical rejection-sampling fine-tuning).

## Why it matters

- **Correct granularity for agentic RLVR.** Flat pass/fail is too coarse; per-step rewards are often unavailable. Edge-level verifiers are the natural intermediate: structured, cheap to author, discriminative at the trajectory level.
- **Shared artefact for task spec and rubric.** The graph produces both the public task description and the private rubric, so they stay in sync — a common failure mode of hand-authored benches.
- **Captures what matters.** Long-horizon failures are typically *consistency* failures (decisions and evidence drift). Edge types directly target the three ways consistency breaks.
- **Agent-as-judge integrates cleanly.** Edge verifiers that need semantic matching (grounding by quote-equivalence) are delegated to an LLM judge; mechanical edges (invoice-total equality) use code.

## Gotchas & tricks

- **Graph authoring is work.** Writing a completion graph is more effort than writing a flat rubric. Templates per task domain (automation, filing, triage) amortise this.
- **Mandatory vs optional edges.** The paper distinguishes mandatory from optional edges; the verifier only rejects on missed mandatory ones. Over-marking as mandatory makes the task unsolvable; under-marking makes it trivial.
- **Grounding edges need quote matching that handles paraphrase.** Pure substring match misses correct references; an LLM judge is almost always required for grounding.
- **Agent-as-judge contamination.** When the judge is the same family as the agent, systematic blind spots emerge. The paper rotates judges across benchmarks.
- **Reset discipline is critical.** The workspace must reset exactly between rollouts or coupling edges leak signal from a prior trajectory.
- **Not a bench, a representation.** Interlock is a *way to write* long-horizon tasks, not a specific benchmark. The three benches reported (AutomationBench, JobBench, Harvey LAB) are instantiations.

## Sources

- Paper: *Interlock: From Local Correctness to Task Completion in Long-Horizon Agents* — Hao, He, Wang, Yang, Zhang, Hao, Gu, Feng, Wu (PKU + HKUST-GZ + Tsinghua + Meituan), 2026 — [arXiv:2610.19686](https://arxiv.org/abs/2610.19686).
- Background: [rlvr](../post-training/rlvr.md) for the verifier-as-reward paradigm Interlock extends.
