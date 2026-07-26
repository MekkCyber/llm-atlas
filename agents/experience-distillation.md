# Experience Distillation

*Depth — internalise in-context-learning gains into model weights without any new environment interaction.*

**TL;DR:** An agent can learn a lot in-context from its own interaction history (very sample-efficient) — but the gain vanishes when the context is evicted. Plain SFT on those trajectories recovers almost none of the gain. **Experience Distillation** treats the ICL setup as a teacher (context → improved policy) and distils it into the base model via context distillation, using only the trajectories already collected. No fresh rollouts.

**Prereqs:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../post-training/on-policy-distillation.md](../post-training/on-policy-distillation.md) · [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

Three ways to turn a batch of agent trajectories into a better model:

| Recipe | Environment cost | Retained gain over ICL |
| --- | --- | --- |
| Vanilla SFT on trajectories | zero | ~3.8% |
| Classical RL from those trajectories as rewards | many-× | matches ICL only with many rollouts |
| **Experience Distillation (this paper)** | zero | ≥ 64.8% |

Experience Distillation formalises the problem as "given a fixed batch of interaction transcripts that already improved this model when in-context, get most of that improvement into the weights, without any new interactions."

## How it works

The mechanism is **context distillation** applied to interaction traces:

1. Collect a set of trajectories $\{\tau_i\}$ that, when placed in the context of $\pi_\theta$, produce an *improved* policy $\pi^{\text{ICL}}_\theta(\cdot \mid \text{prompt}, \tau_{1:i-1})$.
2. Treat $\pi^{\text{ICL}}_\theta$ as the teacher: it is the *same weights* but with the trajectory prefix in-context, which observably improves it.
3. Distil $\pi^{\text{ICL}}_\theta \to \pi_\theta$ by minimising KL between teacher (with trajectories in-context) and student (no trajectories in-context) on the same set of task prompts.
4. Iterate: the newly-updated student can be re-prompted, produce fresh transcripts, and the loop repeats.

No environment step is executed after the initial trajectory collection.

## Why it matters

- **Sample-efficiency in the right currency.** Environment interactions are the scarce resource for real agent training (SWE tasks: minutes each, human rollouts: hours). Experience Distillation converts *free* ICL improvements into *persistent* weight improvements, buying rollouts.
- **Matches RL with ~10× fewer samples.** The paper reports ≥ 9.6× fewer environment samples to match classical RL baselines across 749 SWE tasks + 6 text-adventure games.
- **Explains why SFT on transcripts fails.** SFT on a transcript imitates the token sequence; ICL on the same transcript conditions the whole context, and Experience Distillation asks "what would the student have output *without* that context." That counterfactual is the signal SFT throws away.
- **Composes with any rollout collection scheme.** Rejection-sampled successful trajectories, tool-use traces, human demonstrations — all can serve as the initial pool.

## Gotchas & tricks

- **Teacher and student share weights.** Unlike distillation from a larger teacher, the "teacher" here is the *same model* with a rich context. Gains cap at whatever the in-context version can achieve — this is a compression of ICL, not a source of new capability.
- **Trajectory selection dominates quality.** Bad trajectories in the context produce a worse in-context teacher, hence a worse student. Filter to outcome-successful trajectories (rejection sampling) before distillation.
- **KL target vs SFT target.** The paper uses distributional KL matching, not point SFT — matching the teacher's *distribution* over next tokens matters, especially where the improved policy hedges across several valid actions.
- **The gap between the 3.8% (SFT) and 64.8% (Exp. Dist.) numbers is the load-bearing headline.** It's the counterfactual claim, not just the technique.

## Sources

- Paper: *Sample-Efficient Learning from Agent Experience* — Gou, Tu, Fang, Cai, Rezatofighi — Monash, 2026 — introduces Experience Distillation, evaluates on 749 SWE tasks + 6 text-adventure games.
- Underlying primitive: context distillation (see e.g. Askell et al. 2021 for the closest LLM-era antecedent).
