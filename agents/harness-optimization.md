# Harness optimization

*Depth — improving an LLM agent by editing its prompts / tools / workflow, not its weights.*

**TL;DR:** An agent's behaviour is split between its *weights* (which you train) and its *harness* — the system prompt, available tools, retrieval policy, plan template, and orchestration code. Harness optimization improves the agent by editing the harness in place, using rollouts the deployed agent already has. RHO (2026) is the canonical self-supervised recipe: pick a coreset of hard past tasks, re-roll multiple harness variants in parallel, self-grade by pairwise preference, and apply the best edit. One round lifts SWE-Bench Pro pass rate from 59% to 78% without touching the model.

**Prereqs:** [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [dpo](../post-training/dpo.md) · [delegation-sft](delegation-sft.md) · [dual-role-self-play](dual-role-self-play.md)

---

## What it is

Two paths to a better agent: (1) train new weights — expensive, requires GPUs, ships a new model; (2) edit the harness — cheap, ships with the deployment, can be A/B-tested overnight. Most agent improvement work has focused on (1); harness optimization treats (2) as a first-class loop.

The harness includes everything that ships with the deployed agent *outside* the model weights: the system prompt, the tool roster and schemas, retrieval keys, the planner template, error-recovery rules, the orchestrator. All of these are editable text — they can be modified, evaluated, and rolled back without retraining.

## How it works — RHO

RHO is one round of the loop:

1. **Coreset selection.** From the deployed agent's trajectory log, pick a small, *diverse, hard* set of tasks (typically the ones with low success or high human escalation). Cardinality matters — too few and the signal is noisy, too many and the round is expensive.
2. **Parallel re-rollouts.** For each coreset task, run the agent *N* times under candidate harness variants (variants are proposed by the agent itself or by a generator prompt).
3. **Self-validation per rollout.** Each rollout is scored by the agent's own self-consistency and self-validation checks — no ground-truth labels.
4. **Pairwise self-preference.** For each task, the agent compares pairs of (harness variant, outcome) and picks a winner.
5. **Aggregate and apply.** The variant that wins most pairs across the coreset becomes the new harness. Bad variants are discarded.

The loop is *fully self-supervised*: it uses only past trajectories + the agent's own judgement, no human grading and no external labels.

## Why it matters

- **No training required.** Ships in production overnight; no GPU budget.
- **Closes the deployed-agent improvement loop.** The hard part of agent improvement has been "what do I do with the trajectory log?" RHO is one good answer.
- **Strong baseline for any agent paper.** Anyone claiming a training-based agent improvement now has to compare against the harness-edit-only version of the same change. Many won't beat it.
- **Real numbers:** 59% → 78% pass on SWE-Bench Pro in one round; gains hold across software engineering, technical work, and knowledge work.

## Gotchas & tricks

- **Self-preference is biased.** The agent's pairwise preference reflects its own quirks (including the very mistakes you're trying to fix). RHO mitigates with diversity in variants + many pairs, but it's still a fundamentally weaker signal than ground-truth grading.
- **Coreset is the most important hyperparameter.** A small, well-curated hard set beats a large arbitrary one. Failure-clustering or human-in-the-loop coreset selection helps a lot.
- **Bound the harness changes.** Allow arbitrary system-prompt rewrites and you get reward-hacked harnesses that pass self-preference but ship strange behaviour. Constrain edits to specific slots (tool descriptions, error-recovery rules) or compose only a fixed library of harness patches.
- **Track regressions on a sanity-check suite.** Hard-coreset gains can mask regressions on tasks the coreset didn't represent. Always re-evaluate the new harness on a stable held-out set before rolling out.
- **One round, not many.** The paper reports a single round of RHO. Iterating naively risks drifting the agent away from its baseline through accumulated self-preference noise.

## Sources

- Paper: *Retrospective Harness Optimization: Improving LLM Agents via Self-Preference over Trajectory Rollouts* — Pan et al., CityU HK / MSRA, 2026 — [arXiv 2606.05922](https://arxiv.org/abs/2606.05922).
- Background: rejection-sampling-style coreset selection — see [rejection-sampling](../post-training/rejection-sampling.md).
