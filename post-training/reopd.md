# Replayed-Prefix On-Policy Distillation (ReOPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An off-environment variant of on-policy distillation that reuses **pre-collected teacher trajectories as replayed prefixes**. The student acts at *selected* steps only; the teacher provides dense per-step supervision without executing any new environment interactions. A step-decaying schedule biases sampling toward earlier prefixes, where the teacher's targets are most reliable. Zero tool calls during student training, ≥4× faster per rollout than online OPD, matching or improving accuracy.

**Prereqs:** [on-policy-distillation.md](./on-policy-distillation.md), [_post-training.md](./_post-training.md).
**Related:** [rejection-sampling.md](./rejection-sampling.md) · [_rl.md](./_rl.md) · [long-cot-rl.md](./reasoning/long-cot-rl.md)

---

## What it is

The paper frames multi-turn OPD as a **prefix distribution design** problem. Two competing forces:

- **Student-relevance:** the more student-like the prefix, the more relevant the teacher's correction — but the further the prefix drifts from what the teacher was trained on.
- **Teacher-reliability:** the closer the prefix is to teacher-native trajectories, the more trustworthy the teacher's target — but the less relevant the correction to the student's actual failure modes.

ReOPD calls the resulting trap the **prefix trap** and manages it explicitly with a sampling schedule over replayed teacher prefixes.

## How it works

1. **Collect** teacher trajectories $\{\tau_T^{(i)}\}$ once, up front, in the target environment(s). No further environment interaction is needed after this.
2. **Sample a prefix**: pick a step index $t$ using a **decaying schedule** biased toward early $t$ (early prefixes are still close to the teacher's own state distribution — low shift).
3. **Splice**: for a fraction of steps within the prefix, let the *student* act instead of the teacher (student-on-policy at those positions).
4. **Query** the teacher at every step of the resulting mixed prefix for a per-step target distribution.
5. **Update** the student with a KL/cross-entropy loss against the teacher targets.

Because the collected trajectory set is reusable, one teacher rollout amortizes across many students and many training passes — ReOPD converts expensive online interaction into a reusable offline resource.

## Why it matters

Environment interaction (tool calls, browser fetches, long-running code) dominates the cost of agent post-training. ReOPD eliminates it from the student loop entirely while keeping most of on-policy distillation's benefit. The prefix trap is the right conceptualization of *why* naive OPD variants don't monotonically improve as you push them more on-policy — the reliability side of the tradeoff is easy to miss.

## Gotchas & tricks

- **Step-decay hyperparameter.** The decay rate trades bias (heavy decay = teacher-like prefixes = safe but stale) vs coverage (light decay = student-like prefixes = risky but relevant). Paper uses a simple geometric decay.
- **Teacher trajectory quality.** Since ReOPD never re-collects, teacher-trajectory diversity at collection time bounds the student's asymptotic performance. Sample environments/tasks broadly.
- **Not a replacement for RL.** ReOPD is a distillation loss — it doesn't discover strategies the teacher doesn't already exhibit. Use it as the SFT-stage-replacement in an RL pipeline, not as the RL replacement.

## Sources

- Paper: *Multi-Turn On-Policy Distillation with Prefix Replay* — Liao et al., 2026 — [arXiv:2607.04763](https://arxiv.org/abs/2607.04763)
- Code: [github.com/baohaoliao/ReOPD](https://github.com/baohaoliao/ReOPD)
