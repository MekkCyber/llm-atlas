# Reward-Reliability Ordering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A design principle for multi-stage RL post-training pipelines: **order the stages by how gameable their reward is**, from hardest-verifiable to softest-judge-based. Verifiable rewards (exact match, unit tests, schema checks) go *first* so the policy learns their capabilities without hacking pressure; environment-grounded rewards (code execution, tool responses) go in the *middle*; preference-based / judge-based rewards go *last* as a polish. Reversing the order — RLHF before RLVR — over-fits the policy to preference judges and destroys the verifiable capabilities the later stages would install. Named and documented by Amazon Rufus in the Rufus-Air recipe.

**Prereqs:** [_rl](_rl.md), [_rewards](_rewards.md)
**Related:** [rlvr](rlvr.md), [grpo](grpo.md), [dpo](dpo.md), [rl-prompt-curation](rl-prompt-curation.md), [rufus-air case study](../case-studies/rufus-air.md)

---

## What it is

Post-training a modern LLM often means chaining several RL stages with different reward sources. The Rufus-Air recipe crystallizes an ordering principle that explains why some published pipelines work (Tülu 3, DeepSeek-R1, Rufus-Air) and others collapse: the *reliability* of a reward — how faithfully it tracks the capability you want — should determine when in the pipeline you optimize against it.

## How it works

Group reward sources into three reliability tiers:

1. **Hard verifiable (RLVR-flavored).** Exact-match on math answers, unit-test pass/fail, schema conformance, keyword-based instruction-following checks. Not hackable in any meaningful way — the verifier's decision *is* the ground truth.
2. **Environment-grounded.** Code execution feedback in an agentic loop, tool responses, retrieval-quality metrics. Concrete but noisier; the reward is a *proxy* for capability, not identical to it.
3. **Judge-based.** LLM-as-judge scores, learned reward models, human preference votes. Softest signal; most susceptible to reward hacking and stylistic gaming.

**The ordering rule.** Run tier-1 stages first, then tier-2, then tier-3.

- **Tier 1 first.** Because the reward is exact, the policy cannot hack it — it must acquire the underlying capability. Whatever the policy learns here is genuine and durable.
- **Tier 2 next.** Environment-grounded stages exercise the capabilities installed by tier 1 in longer horizons. Some hacking is possible but bounded by the environment.
- **Tier 3 last.** Judge-based RLHF is the *polish* step: style, refusal calibration, helpfulness tone. Applied to a model whose capabilities are already installed, so preference-optimization pressure can only push style, not core skills.

Reversing the order (RLHF → RLVR) is a known failure mode: the RLHF stage over-fits to the judge's stylistic preferences, and the subsequent RLVR stage either loses capability (KL fights RLHF's drift) or unlearns the polish.

## Why it matters

- **Explains why several open recipes converge on the same order.** Tülu 3, DeepSeek-R1, OLMo 2 Instruct, and Rufus-Air all put verifiable-reward stages before preference-based stages, even though none of them cite each other for the ordering.
- **Actionable planning tool.** Given a set of reward sources, the tier assignment is usually unambiguous, and the ordering falls out.
- **Diagnostic.** If a pipeline is collapsing at the end, check the ordering first before tuning hyperparameters.

## Gotchas & tricks

- Difficulty filtering ([rl-prompt-curation](rl-prompt-curation.md)) is orthogonal and required at every tier — the ordering principle assumes each stage runs on prompts in its productive band.
- Small KL to the pre-stage reference is critical between tier transitions to prevent late stages from undoing earlier ones; Rufus-Air uses stage-specific KL coefficients.
- Some rewards look verifiable but are secretly judge-based (e.g. "does the response contain a certain concept?" checked by an LLM). Reclassify those into tier 3.
- Tier 2 environment rewards are the most heterogeneous — the same "code execution" reward can be tier 1 (unit tests) or tier 2 (repo-level task success) depending on granularity.

## Sources

- Paper: *Rufus-Air: An Open LLM Post-Training Recipe* — Amazon Rufus, 2026 — [arXiv:2609.29421](https://arxiv.org/abs/2609.29421). Names and formalizes the ordering principle as one of four core findings.
- Related recipe: *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* — AI2, 2024 — implements the same ordering without naming it.
- Related recipe: *DeepSeek-R1* — DeepSeek, 2025 — Stage 2 (Reasoning RL, verifiable) precedes Stage 4 (Alignment RL, judge-based).
