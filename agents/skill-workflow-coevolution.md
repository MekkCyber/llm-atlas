# Skill/Workflow Co-Evolution
*Depth — inference-time growth of a reusable skill library, extracted from workflows the agent has just executed.*

**TL;DR:** Standard skill libraries for agents are built offline and never grow from what the agent actually did during deployment. Skill/workflow co-evolution closes the loop *at inference time*: a successful workflow is analyzed, its promising fragments are compiled into named, executable skills, tagged with per-invocation utility statistics, and made available for future tasks. A pruning rule kills consistently-bad skills so the library stays sharp. No weight updates. FlowEvo reaches **85.6% on ALFWorld** with this recipe.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md), [executable-task-synthesis.md](executable-task-synthesis.md), [../case-studies/composer2.md](../case-studies/composer2.md)

---

## What it is

Two orthogonal artifacts that evolve together:

- **Workflows** — episode-level plans the agent constructs at inference to solve a task. Traditionally discarded after the episode.
- **Skills** — named, reusable, executable routines (shell function, tool-invocation template, sub-agent) callable from a workflow.

Co-evolution promotes selected workflow fragments into new skills and lets subsequent workflows call them. Utility statistics (call count, success rate, cost) drive both promotion and pruning.

## How it works

The inference-time loop:

```
skills = initial_library
for task in stream:
    workflow = plan(task, skills)                    # planner sees the current library
    result = execute(workflow, skills)
    if success(result):
        candidates = extract_fragments(workflow)      # frequent, cohesive sub-plans
        for c in candidates:
            if novel(c, skills) and useful(c):
                skills.add(promote(c))                # give it a name and interface
    update_utility_stats(skills, workflow, result)
    skills = prune(skills, min_utility_threshold)
```

Three design choices matter most:

1. **What counts as a fragment?** Contiguous sub-plans that appear across workflows; typed sub-plans (parameterized templates) beat literal sub-plans.
2. **What counts as useful?** Utility is `success_rate × frequency / average_cost` — success alone rewards single-use skills; frequency alone rewards ubiquitous no-ops.
3. **When to prune?** Prune only when a skill has enough calls to be statistically confident. Otherwise the library churns and callers see broken references.

The whole loop is training-free: no gradient updates, no fine-tuning. Improvement compounds through the library, not through the weights.

## Why it matters

An alternative axis to "train a bigger agent." If the agent can persist and refine its own skill library at inference, capability grows monotonically with usage: no data collection, no post-training, no re-deploy. Complements weight-updating agent RL — the library caches high-utility motifs that would otherwise re-derive on every prompt.

Also gives a natural transfer path: skills promoted on one task family often apply to adjacent ones, giving zero-shot lift on new tasks that reuse those motifs.

## Gotchas & tricks

- **Library rot.** Without pruning, useless or stale skills clutter the planner's context and hurt subsequent workflow quality. Age-weighted utility beats raw utility.
- **Skill collision.** Two skills that overlap in intent split call statistics; neither reaches confident promotion. Add a dedup step (name embedding, behavioral fingerprint) at promotion time.
- **Fragment granularity.** Fragments that are too small (single tool calls) never win over primitive tools; too large ones (whole workflows) rarely reuse. Middle grain — 3–10 steps with clear inputs/outputs — carries the value.
- **Cold start.** Empty libraries produce noisy utility stats early. Seed with hand-written primitives or lift from a prior deployment.
- **Failure of the harness ≠ failure of the skill.** Attribute failures to the specific skill call, not to the whole workflow, or every skill decays after one bad episode.

## Sources

- Paper: *FlowEvo: Self-Evolving Agents through the Co-Evolution of Workflows and Executable Skills* — Ren, Yue, Li et al., 2026 — [arXiv:2607.21596](https://arxiv.org/abs/2607.21596).
- Related: *Voyager* (Wang et al., 2023) — earlier lifetime-skill-library pattern for Minecraft agents.
- Related: *Reflexion* (Shinn et al., 2023) — same episodic-improvement instinct, without the persistent library.
