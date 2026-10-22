# Rubric-based RL
*Depth — using natural-language rubric criteria as the reward signal for open-ended RL, with a just-in-time harness that evolves the rubric with the policy.*

**TL;DR:** For open-ended domains (medical, writing, role-play, instruction-following) there is no verifiable ground truth to reward with. Rubric-based RL replaces the scalar reward with a set of natural-language criteria evaluated by a judge model; the policy is trained to satisfy more criteria over time. Static rubrics quickly stop separating responses or stop correlating with real quality as the policy improves, so **HarnessRL** proposes a *just-in-time harness*: an evolver proposes changes to task mix, rollout guidance, and the live rubric criteria whenever reward separation or quality tracking breaks down, and a critic accepts or rejects each change by measuring its effect on the current policy.

**Prereqs:** [_rl](_rl.md), [_rewards](_rewards.md), [rlvr](rlvr.md), [rl-prompt-curation](rl-prompt-curation.md)
**Related:** [grpo](grpo.md), [cot-reward-model](cot-reward-model.md), [reasoning/self-credit](reasoning/self-credit.md)

---

## What it is

An alternative to [RLVR](rlvr.md) for domains without rule-checkable answers. The reward comes from a *rubric*: a structured list of natural-language criteria (e.g., "cites specific authorities", "distinguishes dosage by patient class", "flags contraindications before recommending"), each scored 0/1 by a judge model and summed into a scalar.

## How it works

**Rubric as reward.** For prompt $p$ and response $y$:

$$ R(p, y) = \sum_{c \in \mathrm{rubric}(p)} \mathrm{judge}(c, p, y). $$

**Two failure modes.** Rubrics decay in two independent ways as the policy improves:
- *Separation collapse*: all responses satisfy all criteria, so rewards cluster near max and advantages flatten.
- *Quality drift*: the judge keeps separating responses, but the direction of separation stops correlating with real quality (the policy is gaming the judge's observable surface).

**Just-in-time harness (HarnessRL).** Three evolvable levers are tracked:
1. **Task mix** — which prompts to sample next.
2. **Rollout guidance** — the system prompt / planning scaffold the policy uses while generating.
3. **Rubric slice** — which criteria from the full catalog are live for the current prompt.

An **evolver** diagnoses each failure (separation collapse vs quality drift) and proposes a change to one lever. A **critic** admits the change only after running a short rollout to measure its effect on reward separation *and* on an auxiliary quality probe (held-out pairwise comparison, or a strong external judge). The accepted change becomes the next step's training config.

**Dataset.** Rubric-Mix-10k spans medical, science, writing, role-play, and instruction-following prompts, each with a hand-curated full rubric catalog plus a default live slice.

## Why it matters

- **RLVR without verifiable answers.** Opens open-ended domains to the same group-relative RL machinery used for math/code.
- **Keeps the reward signal useful throughout training.** Fixed-rubric RLHF plateaus once the policy adapts to the judge; the just-in-time harness pushes the problem forward instead of letting it stall.
- **Separates reward engineering from policy optimization.** The evolver/critic loop concentrates reward-design iteration into a measurable sub-problem.

Reports gains over the strongest baselines at 4B and 9B scales on Rubric-Mix-10k, reaching the best checkpoint in roughly half the training steps.

## Gotchas & tricks

- **Judge drift.** Using the policy itself as the judge is cheap but creates feedback loops; a frozen third-party judge is safer.
- **Separation ≠ quality.** The critic must probe both axes; otherwise the evolver happily hill-climbs on separation and drives the policy into judge-gaming behaviors.
- **Rubric catalog is the real spec.** The quality ceiling is set by the catalog; the harness only picks slices from what's already there.
- **Overlapping criteria.** Highly-correlated criteria don't add signal; the evolver should prefer orthogonal slices.
- **Prompt curation reaches diminishing returns alone.** The win versus pure prompt-curation baselines comes from the *rubric slice* lever — turning it off recovers most of the gap.
- **Not for safety-critical rewards.** Judge models hallucinate; rubric-based rewards should not be the sole gate on refusal / dangerous-capability training.

## Sources

- Paper: *HarnessRL: Co-Evolving Policies with a Just-in-Time Training Harness for Open-Ended RL* — Alibaba Token Hub + Mila, 2026 — [arXiv:2610.26426](https://arxiv.org/abs/2610.26426).
