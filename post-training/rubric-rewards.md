# Rubric Rewards
*Depth — explicit, criterion-wise reward signals combined with a learned preference reward for post-training (especially T2I).*

**TL;DR:** Preference reward models learn aesthetics/style well but reward-hack on prompt adherence; rubric rewards do the opposite. **Rubric rewards** are scored by an LLM judge against a *written criterion list* — prompt faithfulness, constraint satisfaction, reward-hack checks — and are composed with a Bradley–Terry preference RM via a weighting that is tuned on arena data. Introduced as part of a frontier T2I post-training recipe (2026) that reports Elo gains on the T2I arena.

**Prereqs:** [_rewards.md](_rewards.md), [_rl.md](_rl.md)
**Related:** [dpo.md](dpo.md) · [grpo.md](grpo.md) · [cot-reward-model.md](cot-reward-model.md)

---

## What it is

A reward design pattern that externalises what a learned RM gets wrong. Instead of "one preference RM to rule them all," the pipeline scores each generation under two signals in parallel:

- **Preference reward.** Bradley–Terry model trained on pairwise human votes (arena-style). Learns the global quality signal but is prone to reward hacking on things humans weren't shown.
- **Rubric reward.** An LLM judge scores a written rubric: *"does the generation faithfully depict all nouns in the prompt?"*, *"does it respect the constraint 'no text'?"*, *"are any of the common reward-hack patterns present (over-smoothing, unnatural saturation, cliché composition)?"*

A composition stage weights the two into a single scalar per generation, feeding the usual RL post-training (PPO/GRPO) or DPO loop.

## How it works

- **Rubric authorship.** A small set of criteria (5–20) tied to known failure modes of the preference RM. Each criterion gets a short LLM-judge prompt that produces a 0/1 or 0–5 score.
- **Preference RM.** Standard Bradley–Terry on arena pairs.
- **Composition.** Linear (or sigmoid-gated) combination of normalised rubric scores with the preference reward; weights are picked on held-out arena data to maximise correlation with human-labelled hack-free wins.
- **Reward-hack detection as a rubric.** A notable move: include "is this output exhibiting known reward-hack patterns?" as its own rubric criterion, so the composed reward actively penalises the preference RM's own failure cases.

The composed reward then drives standard RL post-training.

## Why it matters

T2I RLHF has drifted from monolithic preference RMs to layered signals — exactly the trajectory text RLHF followed a year earlier. Rubric rewards give a reproducible recipe for that layering and name the failure modes concretely. The pattern generalises:

- **Code.** Preference RM on human-labelled solutions + rubric on test pass/style/complexity.
- **Reasoning.** Preference RM on CoT quality + rubric on step verifiability.
- **Agents.** Preference RM on task success + rubric on safety / resource / audit criteria.

## Gotchas & tricks

- Rubric judges can be reward-hacked too — mix multiple judge models or periodic human spot-checks.
- Keep the rubric *short*. Long criterion lists dilute the signal and are expensive to score at RL scale.
- Composition weights drift over training as the policy learns to satisfy some criteria easily; retune mid-training.
- A rubric that encodes "no reward hacking" is only as strong as the hack patterns listed. Treat the list as a living artefact and update it when new hacks appear.
- Rubric scoring at RL batch sizes is expensive; cache judge outputs on identical generations and consider a smaller distilled rubric judge.

## Sources

- Paper: *Post-Training Frontier Text-to-Image Models by Composing Preference and Rubric Rewards* — Hsu, Angelopoulos, Chiang, Stoica, Hsieh — Arena Intelligence / UCLA, 2026 — [arXiv:2610.02967](https://arxiv.org/abs/2610.02967).
