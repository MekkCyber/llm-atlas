# Rubric-as-Reward
*Depth — synthesize a hierarchical rubric per instruction, then reuse the same rubric to filter SFT data and to score RL rollouts.*

**TL;DR:** Instead of training a scalar reward model, generate a *hierarchical rubric* per instruction up front — a short tree of criteria (and sub-criteria) the response must satisfy — and use that rubric both to filter SFT data (keep only responses that score above threshold) and to score rollouts during RL. One decomposition is reused for two purposes: data curation and reward. Introduced for multimodal code generation in PrismaCoder, where each rubric mixes code-level and visual-level criteria.

**Prereqs:** [rlvr](rlvr.md), [_rewards](_rewards.md)
**Related:** [rejection-sampling](rejection-sampling.md), [cot-reward-model](cot-reward-model.md), [../data/quality-filtering](../data/quality-filtering.md)

---

## What it is

Scalar reward models (RMs) collapse many quality axes into one number trained on human preferences. Rubric-as-reward keeps the axes *separate and legible*:

- For each instruction, a strong model generates a rubric with explicit, named criteria.
- Each criterion is checkable — either by code, by executing the response, or by a judge LLM reading the rubric.
- The response's score is the aggregation (sum, mean, weighted) over criteria.

The same rubric is then used in two places:

1. **SFT filtering.** Candidate responses are scored against the rubric; only those above a threshold go into the SFT set.
2. **RL reward.** During RL rollouts, the rubric is the reward function (binary-per-criterion or weighted scalar).

Because the rubric is per-instruction, the "reward model" is implicitly conditioned on the task — avoiding one of the biggest failure modes of generic preference RMs, which average across tasks.

---

## How it works

### Rubric synthesis

For instruction *p*, a frontier model generates a hierarchical rubric:

```
rubric(p) = {
  code_criteria: [c1, c2, ...],
  visual_criteria: [v1, v2, ...],   # when outputs are visual (chart, web)
  ...
}
```

Each criterion is phrased as a yes/no or 1–5 question. For code-visual tasks (chart generation, web pages, animations), rubrics split by modality: *code-level* (compiles, uses the requested library, has correct structure) and *visual-level* (matches target description, follows the layout spec).

### Dual use

- **Data pipeline.** Candidate completions are scored; only `score > τ` enters the SFT corpus. This filters orders of magnitude more aggressively than keyword/length heuristics and more cheaply than human review.
- **RL reward.** For each rollout, the same scoring function returns a scalar (or per-criterion vector) that the policy optimizer consumes. PPO/GRPO work unchanged.

Because the rubric was already used to filter SFT data, the SFT model is biased toward producing responses the rubric scores highly, which reduces mode mismatch between SFT and RL — a common cause of RL collapse.

---

## Why it matters

Three gains over scalar RMs:
1. **Per-task specificity** — a rubric for a chart task scores chart criteria, not generic "helpfulness."
2. **Legibility** — failures surface as "criterion *v3* always fails," enabling targeted debugging; scalar RMs give one number.
3. **One decomposition, two uses** — filtering and reward come from the same source, closing a mismatch that scalar RM pipelines routinely open.

The practical downside is cost: rubric generation is model calls per instruction, not amortized across the dataset.

---

## Gotchas & tricks

- **Rubric drift.** If rubrics are re-generated between SFT and RL (e.g. different model, different prompt), the two signals diverge; snapshot the rubric set once and reuse.
- **Criterion weights.** Equal weights are a reasonable default; task-specific weights (code-weight vs. visual-weight) sometimes help but introduce a tuning knob.
- **Judge-LLM failure modes.** When criteria are too subjective, judges disagree with each other more than with the policy; keep criteria concrete and executable where possible.
- **Not a free lunch for open-ended tasks.** Rubric synthesis works best when the response has a verifiable structure. For "write a poem," rubric criteria devolve into vague judge calls.

---

## Sources

- Paper: *PrismaCoder: Pushing the Boundaries of Open Multimodal Code Intelligence through Decomposed Rubric Synthesis* — PrismaCoder team, HKU, 2026 — [arXiv:2610.20538](https://arxiv.org/abs/2610.20538).
- Related: RLHF reward-model literature (Christiano et al., 2017; InstructGPT, 2022); constitutional-AI critique prompts (Anthropic, 2022).
