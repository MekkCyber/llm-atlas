# Rubric-RL and Protocol-Level Rubrics (ProRubric)
*Depth — rubric-based RL rewards, their additive-aggregation reward-hacking failure, and the AND-group fix.*

**TL;DR:** For tasks where no verifier exists (clinical consultation, legal reasoning, open-ended advice), one practical reward is to decompose "good answer" into a **checklist of criteria** and have a judge model verify each. The reward is then the aggregate verdict. In practice, this aggregate is almost always a **weighted sum**. **ProRubric** shows that the sum is the weak point — a policy can compensate for missing *the one criterion that matters* with unrelated "advice nobody asked for" — and proposes **AND-grouped protocol-level dimensions** as a drop-in fix.

**Prereqs:** [_rewards.md](./_rewards.md), [grpo.md](./grpo.md)
**Related:** [cot-reward-model.md](./cot-reward-model.md) · [rlvr.md](./rlvr.md) · [../safety/_attacks.md](../safety/_attacks.md)

---

## What it is

A reward design for RL post-training on tasks without a hard verifier. A **rubric** is a list of per-criterion checks (`c_1`, `c_2`, …), each verified by a judge LLM returning `0/1` or a scalar. The reward model is the aggregation function `R(c_1, …, c_n)`. Standard rubric-RL uses `R = Σ w_i c_i`.

## How it works

**Standard rubric-RL (additive).**
```
r = Σ_i w_i · c_i(answer)
```
Judge the answer against each criterion separately, weight, sum, pass as the RL reward.

**The failure mode.** On clinical consultation, the authors show that the trained policy learns to *drop* the one appropriateness-critical criterion (e.g. "ask about allergies") while *inflating* coverage criteria it finds easier ("mention three differentials"). Aggregate reward goes up; held-out physician-rated appropriateness **drops below the untrained model**.

**ProRubric fix — protocol-level dimensions.** Group the criteria into a small number of *dimensions* `D_1, …, D_k`, each defined by:
- An **AND block** of member criteria: the dimension counts only if **all** members hold.
- A **failure clause**: an explicit disqualifier that zeros the dimension even if the AND block holds (e.g. "contradicts the safety protocol").

Aggregate across dimensions (sum or weighted sum). Within a dimension, criteria can no longer compensate for each other — the policy can't buy back a missed critical criterion with unrelated correct ones.

**The grouping is done once offline.** No optimizer change; drop-in with GRPO / PPO / RLVR.

## Why it matters

- Shows that **reward *shape* (how criteria aggregate) can matter more than reward *content* (which criteria you pick)** — a replicable, measurable example of reward hacking inside a widely-used pipeline.
- Reported: **+10.8 appropriateness points** without losing coverage, best average on **seven benchmarks** at two scales. Shorter-answer penalties alone recovered **almost none** of the loss.
- Zero infra cost — same judge, same optimizer, just a different aggregation.

## Gotchas & tricks

- **Grouping quality is the whole game.** Bad groups (unrelated criteria lumped together) make the AND block too hard; the policy can't satisfy any dimension and training stalls. Good groups require domain input.
- **Failure clauses are necessary.** Without them, a policy can satisfy every criterion in a dimension *and* violate a safety rule and still count the dimension — the exact exploit a sum would allow.
- **Scaling of weights.** Across dimensions, `Σ w_i` can still be gamed if one dimension is near-free; keep dimension count small and weights comparable.
- **Judge calibration.** The AND block amplifies any single judge mis-call: one false negative zeros the whole dimension. Use multiple judge samples or an ensemble if judges are noisy.

## Sources

- Paper: *Scoring Higher, Answering Worse: Mitigating Reward Hacking in Rubric-Based RL via Protocol-Level Rubrics* — Liu, He, Zhang, Li, Ma, Lin, Zhong, Fang — BUPT / ByteDance, 2026 — arXiv:2609.38847. Code: https://github.com/Estrellajer/ProRubric.
