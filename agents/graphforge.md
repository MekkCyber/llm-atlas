# GraphForge
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An agent-training data-synthesis pipeline that grounds both the task statement and its verification rubric in *real* files. For each occupation-grounded seed, GraphForge assembles a workspace of real files, builds an evidence graph over their relations, and derives rubric criteria from graph edges so every requirement is traceable back to specific files on disk.

**Prereqs:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)
**Related:** [../post-training/_rewards.md](../post-training/_rewards.md), [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

Working agents (OpenHands-, Claude-Code-style) need to open many real files, run tools, and produce deliverables. Training data for such agents has historically come either from (a) model-generated synthetic files (unrealistic, low diversity) or (b) real-file tasks with no task-specific verifier (unchecked quality). GraphForge (Su et al., Fudan / USTC, 2026) fixes both failure modes by letting an **evidence graph over real files** generate the task *and* the rubric.

## How it works

1. **Occupation-grounded seeding.** Diverse seeds drawn from a catalogue of occupational workflows provide controlled task diversity.
2. **Workspace assembly.** For each seed, a real set of files (spreadsheets, documents, code, data) is gathered to populate a working directory.
3. **Evidence graph.** Nodes = files, edges = relations (citation, dependency, mention). The graph structure is the ground truth about what-depends-on-what in the workspace.
4. **Task + rubric from the graph.** A task statement and its per-criterion rubric are both generated against the graph, so every rubric item is anchored to the specific files needed to verify it.
5. **Executability check.** An initial rollout verifies the task is actually executable; a revision agent repairs task/rubric misalignments against the original files before trajectories are kept.
6. **Rejection fine-tuning.** After an initial SFT pass, further rollouts are selected using the same evidence-anchored rubric as the selection signal.

## Why it matters

- Converts "we need realistic multi-file agent trajectories" into a programmable pipeline.
- The rubric is a reusable artifact — it can both score training data and serve as an inference-time check.
- Fine-tuning Qwen3.6-27B on **2,169 GraphForge trajectories** lifts GDPVal under OpenHands by **+65.7** (to 1445.7), Workspace-Bench-Lite by **+7.7**, SpreadsheetBench II by **+13.7** under Claude Code. Rubric-selected rejection fine-tuning compounds the gains.
- Rubric-as-selection-signal is a cleaner alternative to generic LM-judge filtering for agent data.

## Gotchas & tricks

- Realism depends on the seed catalogue; occupation taxonomy quality propagates to task diversity.
- Executability check and revision agent are load-bearing — removing either lets degenerate tasks into the trajectory set.
- The evidence graph is only as useful as its edges — if citation/dependency extraction misses relations, the rubric criteria are not fully anchored.
- Rubric scoring is scalar per criterion; combining criteria with equal weight can be too coarse for tasks with one critical and several secondary deliverables.

## Sources

- Paper: *GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis* — Su et al., Fudan / USTC, 2026 — [arXiv:2609.38923](https://arxiv.org/abs/2609.38923).
- Benchmarks referenced: GDPVal, Workspace-Bench-Lite, SpreadsheetBench II.
- Context: *OpenHands*, *Claude Code* — the two agent harnesses the SFT'd model was evaluated under.
