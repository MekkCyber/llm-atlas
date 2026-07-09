# Turn-aware on-policy distillation (TurnOPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Vanilla on-policy distillation (student mimics a teacher on the student's own trajectories) wastes compute on long-horizon agent tasks in two ways: (1) full-horizon rollouts spend wall-clock on tail turns that produce weak, noisy KL signal; (2) trajectory-level KL loads most of the loss on shallow (early) tokens, so deeper decision turns get under-trained. TurnOPD introduces a **turn-level budgeting** strategy — allocate rollout compute and KL loss per turn, not per trajectory — fixing both.

**Prereqs:** [_rl.md](./_rl.md), [_post-training.md](./_post-training.md)
**Related:** [grpo.md](./grpo.md), [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md), [../agents/README.md](../agents/README.md)

---

## What it is

On-policy distillation (OPD) trains a student policy by matching a stronger teacher on the *student's own* trajectories — cheaper than full RL because rewards are replaced by teacher-KL. Attractive for long-horizon agent training where reward signals are sparse and RL rollouts are expensive.

But naïvely-transferred OPD hits two inefficiencies on long-horizon agent tasks:

1. **Wasted tail-turn compute.** Late turns in a long trajectory often diverge to states where the teacher's KL is uninformative (either the teacher has similar low confidence, or the trajectory is already off the useful manifold). Yet full-horizon rollouts pay wall-clock for them.
2. **Shallow-token KL concentration.** Trajectory-level KL sums per-token KL across turns. Early tokens have high absolute KL (easily distinguished distributions); deeper turns have small absolute KL (well-aligned or narrow choices), so their share of the loss is negligible — under-trained.

TurnOPD introduces a **turn-level budgeting strategy** to reallocate compute and loss.

## How it works

Two mechanisms:

**1. Tail-turn budget.** Cap or skip rollouts of tail turns where expected KL informativeness is low. Save wall clock; reallocate to early / mid turns that produce useful supervision. The cap can be static (rollout up to $T$ turns) or adaptive (rollout while informativeness > threshold).

**2. Turn-aware KL weighting.** Instead of summing per-token KL over the whole trajectory, weight each turn's contribution so deeper turns get proportional signal. Simple form: normalize KL loss per turn, then average across turns. This forces the deeper decision points to influence the update.

Both mechanisms plug into any standard OPD loop (student rollouts, teacher forward passes, KL loss).

## Why it matters

- **Efficient long-horizon on-policy distillation** at Tencent Hunyuan scale — long-horizon agent training was previously OPD-inefficient enough that RL was preferred despite its cost.
- **Reframes the unit of compute** in agent RL/distillation. Trajectory-level thinking hides where the training signal actually lives (per turn). Once you look per-turn, the fixes become obvious.
- **Complements agent RL.** OPD needs a teacher; TurnOPD makes OPD cheap enough that "distill from a stronger checkpoint of the same family" becomes a competitive alternative to end-to-end RL for agents.

## Gotchas & tricks

- **Turn boundaries need to be well-defined.** In multi-turn dialog, "turn" is the user/assistant boundary. In tool-using agents, "turn" is a tool-call round. Skip granularity matters — token-level weighting alone doesn't help without turn boundaries.
- **Tail-turn skipping trades bias for variance.** You lose signal from actually-informative tail turns. In-domain adaptation: if the agent's failure mode is late-trajectory (e.g., forgetting instructions), tail turns *are* the signal.
- **The teacher needs to work well on the student's states.** OPD assumes the teacher can produce useful supervision on student trajectories. If the student's rollout distribution diverges too far from the teacher's training distribution, KL becomes uninformative for a different reason.
- **Not a substitute for RL.** For tasks where reward signal exists (verifiable outcomes, tool-call success), full RL on the same rollouts can beat OPD. TurnOPD is best when you have a stronger teacher and no cheap reward.
- **Compatible with any KL-based distillation loss** — token-level KL, sequence-level KL, or hybrid. The budgeting is orthogonal.

## Sources

- Paper: *TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training* — Zhou, Zheng, Li, Peng, Xu, Chen, Tencent Hunyuan, 2026 — [arXiv:2607.05804](https://arxiv.org/abs/2607.05804).
