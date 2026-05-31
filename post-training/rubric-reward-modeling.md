# Rubric Reward Modeling (RUBRIC-ARROW)

*Depth — rubric-based pointwise rewards trained without frontier-LLM rubric generation.*

**TL;DR:** Rubric methods decompose subjective scoring into explicit criteria, then aggregate. The standard recipe relies on a frontier LLM to draft rubrics and uses hard-Boolean aggregation, producing many tied scores. RUBRIC-ARROW (1) jointly trains a *rubric generator* and a *rubric-conditioned judge*, alternating via GRPO, and (2) replaces Boolean aggregation with a probability-based scoring rule that rarely ties. Only pairwise preference data is needed for the RL stage; no frontier teacher.

**Prereqs:** [grpo.md](grpo.md), [_rewards.md](_rewards.md)
**Related:** [cot-reward-model.md](cot-reward-model.md), [reasoning/prm.md](reasoning/prm.md), [rlvr.md](rlvr.md)

---

## What it is

A reward-modeling recipe for non-verifiable domains (helpfulness, creative quality, multi-aspect writing) where rule verifiers don't apply. Rubric methods stand between RLVR (no learned RM, only works where verifiers exist) and preference RMs (most hackable; coarse). The rubric is itself the "verifiable" surface — once it exists, judging it is structured.

## How it works

Two models trained jointly:

1. **Rubric generator** $G$ — given a prompt, emit a structured rubric (a list of criteria with weights).
2. **Rubric-conditioned judge** $J$ — given (prompt, response, rubric), produce a scalar pointwise score.

The RL loop alternates GRPO updates of $G$ and $J$, each under preference-pair rewards (the model whose judgment correctly ranks pairs is preferred):

- Update $J$ while $G$ is frozen — $J$ learns to score under the current rubric distribution.
- Update $G$ while $J$ is frozen — $G$ learns to emit rubrics that let $J$ rank pairs accurately.

The scoring rule for $J$ is probability-based, not Boolean: each criterion's "pass" emits a probability in $[0, 1]$ rather than $\{0, 1\}$, and the aggregate is the (weighted) sum of probabilities. This avoids the discrete-tie problem of "criteria 3 of 5 pass = same as criteria 3 of 5 pass." Two responses can now differ by fractional probabilities even when they pass the same number of criteria.

Only pairwise preference data is required to train the RL stage — no per-criterion human labels.

## Why it matters

- Pushes a rule-like structured signal into domains where rule verifiers don't apply. The rubric is the verifiable surface.
- Removes the frontier-LLM dependency that current rubric pipelines have for rubric generation, which is the cost / accessibility blocker for rubric methods at scale.
- Probability-based aggregation gives RL gradients real signal where Boolean aggregation produces flat regions of tied rewards (gradient = 0).

## Gotchas & tricks

- Alternating optimization can collapse: $G$ emits trivially easy rubrics, $J$ scores everything as perfect. Preference-pair rewards counter this — a rubric that lets every candidate score the same can't rank pairs. Watch the RL gradient norm of $G$; if it goes near zero, raise the entropy on preference-pair selection.
- The judge's CoT (if it uses one) inherits the cost story of [cot-reward-model.md](cot-reward-model.md) — multiplied by the number of candidates per prompt.
- Rubric-conditioned scoring is more transparent than a black-box RM; the per-criterion probabilities provide built-in explanations of why a response scored what it did. Useful for debugging RM behavior.

## Sources

- Paper: *RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains* — 2026 — [arXiv 2605.29156](https://arxiv.org/abs/2605.29156).
