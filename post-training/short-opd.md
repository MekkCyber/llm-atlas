# ShortOPD (Short-to-Long On-Policy Distillation)
*Depth — an adaptive-horizon OPD schedule that skips wasteful long rollouts early and grows the horizon as the student can use it.*

**TL;DR:** After structured pruning, generation collapses primarily through **suffix repetition** — useful prefixes are still there but the tail degenerates. Vanilla on-policy distillation (OPD) fixes this but wastes early-training compute on long, low-information repetitive tails. **ShortOPD** (Yuan et al., 2026) detects teacher-confirmed repetitive suffixes, treats each rollout's *effective length* as the surviving prefix, and allocates future rollout budgets to the lengths the student can currently use. Reaches ~**9× baseline post-compression score**, beats standard recovery recipes (SFT-only, KD, SeqKD) by **1.6–4.4×**, and matches an 8192-token OPD baseline within 2 points using **a quarter of the training time** (8.5 vs 35.9 hours) and **71% fewer rollout tokens**.

**Prereqs:** [on-policy-distillation](on-policy-distillation.md)
**Related:** [_post-training](_post-training.md)

---

## What it is

A short-to-long **rollout scheduling** modification of standard OPD, designed for the specific setting of recovering generation quality in a compressed (structurally pruned) LLM using its pre-compression checkpoint as the frozen teacher.

Two grounded observations motivate it:

1. After structured pruning, greedy **pass@1** nearly vanishes even while **pass@k** under sampling remains useful — the failure mode is *demotion*, not erasure.
2. The recoverable regime fails **mainly through suffix repetition** — a decent prefix followed by degenerate loop of a few tokens.

Consequence: long OPD rollouts on the pruned model spend most of the training tokens on degenerate tails that carry almost no gradient signal.

## How it works

**Standard OPD (baseline):** rollout the pruned student for a fixed horizon (e.g., 8192 tokens); query teacher for per-token distribution across the whole rollout; distill.

**ShortOPD:**

1. **Detect the repetitive suffix per rollout.** Use the teacher's own signal to identify where the student's output enters degenerate loop — a "teacher-confirmed" repetition marker (e.g., high-probability short cycles the teacher would not have produced).
2. **Treat the surviving prefix as that rollout's effective length.** Truncate to the effective prefix; discard the degenerate tail from the loss.
3. **Allocate future rollout budget to the lengths the policy can currently use.** As training progresses and the student's effective prefix gets longer, expand the horizon — but only in step with capability. This is the "short-to-long" schedule.

The rollout budget stays roughly constant, but nearly all of it is spent on informative tokens.

## Why it matters

- **Structured pruning has been over-validated on multiple-choice and under-validated on generation.** ShortOPD makes recovery cheap enough that generation quality can be re-measured routinely after compression, closing a chronic deployment gap.
- **Concrete productivity wins.** 4× wall-clock speedup and 71% fewer rollout tokens vs. fixed 8192-horizon OPD — with matching quality. Real production teams can adopt this.
- **Reframes OPD from a luxury to a primitive.** OPD used to be considered expensive vs. offline KD; ShortOPD's efficiency undercuts the argument, making OPD a default in compression pipelines.
- **The short-to-long insight generalizes.** The idea of *effective* rollout length adapting to policy capability applies beyond pruning recovery — early-training RL and OPD on new domains face the same waste pattern.

## Gotchas & tricks

- **Repetitive-suffix detector is load-bearing.** Too aggressive → drops legitimate long generations; too lax → keeps the degenerate tails. Calibrate on a held-out set of pre-compression rollouts.
- **Effective-length distribution shifts over training.** Early on, most rollouts are short; late, most are long. Log the distribution as a training-health metric.
- **Budget allocation strategy is a knob.** The paper's approach — allocate future budget to lengths the policy uses — is one option; capacity-proportional or uniform-over-capable-lengths are alternatives worth ablating.
- **Not a substitute for outcome verification.** ShortOPD trains on teacher-confirmed prefixes but doesn't verify final-answer correctness. For math/code recovery, combine with reject-sampling on correct completions.
- **Reported gains are for pruning recovery.** For clean-teacher-to-clean-student distillation (no compression involved), the short-to-long dynamic still exists but the gains are smaller.

## Sources

- Paper: *ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation* — Yuan, Lin, Lu, Han, Sun, Li, Xu, Li, Zhao, 2026 — [arXiv 2607.13124](https://arxiv.org/abs/2607.13124). ByteDance + Institute of Software, CAS + UCAS.
