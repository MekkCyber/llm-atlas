# MetaRubric — Counterfactual Rubric-Based RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Rubric-based RL scores a response against a list of criteria using a judge LLM — but the judge often awards credit even when the required evidence is absent. The paper names this *vacuous credit*. MetaRubric alternates evidence-aware policy optimization (credit only if the response contains sufficient evidence under this prompt's facts) with response-guided rubric adaptation (criteria rewritten against counterfactual prompts). +6.0 to +20.4 pp on PubMedQA over static-judge GRPO. Introduced in *MetaRubric* (2026, arXiv:2610.02824).

**Prereqs:** [_rewards.md](_rewards.md), [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [cot-reward-model.md](cot-reward-model.md), [rlvr.md](rlvr.md)

---

## What it is

Rubric-based RL extends RLVR to open-ended domains where rule verifiers don't apply (medicine, law). A rubric is a list of criteria the response should satisfy; a judge LLM assigns partial credit per criterion. The failure mode: judges award credit when a response "sounds right" even if the actual evidence for a criterion is missing — "vacuous credit." That credit persists after the information is removed, and can flip the sign of a response's GRPO advantage.

## How it works

Two coupled passes alternate:

### 1. Evidence-aware policy optimization

For each prompt, construct a **counterfactual** by changing one task-relevant fact. The judge is asked to grade both the original and counterfactual responses against the rubric. A criterion only earns credit if the response's evidence suffices for the criterion *under that prompt's facts* — detected by the credit *differing* between the original and counterfactual pair.

Formally, credit for criterion $c$ on prompt $p$ with response $r$:
$$
\mathrm{credit}(c, p, r) = \mathbb{1}[j(c, p, r) = \text{satisfied}] \cdot \mathbb{1}[j(c, p', r') \ne j(c, p, r)]
$$
where $(p', r')$ is the counterfactual pair for the same criterion. GRPO advantages are computed on these evidence-filtered credits.

### 2. Response-guided rubric adaptation

After each policy-optimization stage, the current policy's responses guide revisions to the criteria. Both original and counterfactual criteria are rewritten (preserving the original prompt's intended meaning) to track observed policy errors. Criterion weights are adapted at stage boundaries to focus on where the policy is weakest.

## Why it matters

Rubric judges are increasingly used for RL in domains where RLVR doesn't apply, so a baked-in judge failure propagates everywhere. The counterfactual-pair construction is a cheap mechanical fix that doesn't require retraining the judge. Reported: +6.0 to +20.4 pp on PubMedQA across multiple backbones vs static-judge GRPO; further gains on HealthBench-Hard and two multimodal medical benchmarks.

## Gotchas & tricks

- **Counterfactual construction is the hard part.** The "flip one fact" needs to flip a *task-relevant* fact without breaking the question's coherence. The paper uses a seeded template approach; freeform generation gave weaker counterfactuals.
- **Judge still gets gamed over time.** Policies learn to produce evidence-shaped text that satisfies the counterfactual check without being truly grounded. Rubric adaptation partially fights this.
- **Costs ~2× per rollout.** Each prompt is scored twice (original + counterfactual). At scale this is substantial; the paper argues the quality gain more than pays for it.
- **Weight adaptation is coarse.** Per-criterion weights are updated at stage boundaries, not per-step. Finer adaptation was unstable.

## Sources

- Paper: *MetaRubric: Learning to Reward for Rubric-Based Reinforcement Learning* — Fan, Yoon, 2026 — [arXiv:2610.02824](https://arxiv.org/abs/2610.02824).
