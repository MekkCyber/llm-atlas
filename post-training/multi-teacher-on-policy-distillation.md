# Multi-Teacher On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train multiple **domain-specialist teachers** independently (typically each via its own RL run against its own reward signal), then consolidate all specialists plus a shared base into a single **unified student** via on-policy distillation. Each specialist teacher provides supervision on its domain; the student's rollouts are scored against each teacher on the tokens where that teacher is authoritative. Introduced (in this formalized recipe) by Motif 3 (2026) as the post-training pipeline for its 314B / 13.2B-active MoE.

**Prereqs:** [on-policy-distillation.md](on-policy-distillation.md), [grpo.md](grpo.md), [rlvr.md](rlvr.md)
**Related:** [u-opsd.md](u-opsd.md), [spot-distillation.md](spot-distillation.md), [../case-studies/motif-3.md](../case-studies/motif-3.md)

---

## What it is

DeepSeek-V3 popularized an ad-hoc version of this recipe: train a reasoning specialist (R1) with RL, then distill its long-CoT traces back into the frontier base via SFT. Multi-teacher on-policy distillation generalizes the idea to a **portfolio of specialists** — one per capability domain (math, code, science, instruction-following, safety, tool use, etc.) — merged into one student via OPD instead of SFT.

The "on-policy" part matters: the student samples completions, and each teacher provides per-token distributions on the student's own rollouts (not on teacher rollouts, which would be off-policy for the student). The "multi-teacher" part means the target distribution at each token is a **teacher-weighted mixture**, where weights depend on which specialist is most authoritative for the current context.

## How it works

**Stage 1 — Train specialists independently.** For each domain `d ∈ {1, …, D}`:
- Start from a shared SFT'd base.
- Run RL (typically GRPO with domain-specific verifiers) to specialize.
- The result: `D` specialist checkpoints, each strong at its domain but drifted from the general base elsewhere.

Motif 3's specific setup: six RL specialists plus one SFT-only software-engineering specialist, total D = 7.

**Stage 2 — Multi-teacher OPD.** With the shared base (or general-SFT'd base) as the initial student:
1. **Rollout.** Student generates completions on a mixture of prompts spanning all domains.
2. **Teacher scoring.** For each token in a rollout, run each specialist teacher to get its per-token distribution.
3. **Authority weighting.** Compute per-token weights `w_d(t)` indicating which teacher is authoritative — commonly a soft router based on prompt embedding, or a hard router by prompt domain. In practice a hybrid: hard-route on obvious domains, soft-route on ambiguous prompts.
4. **Distill.** Target distribution at token `t` is `Σ_d w_d(t) · π_{teacher,d}(· | context)`. Student is updated by per-token KL to this mixture.
5. **Iterate.** Fresh rollouts each step.

The KL update pulls the student toward each teacher exactly on the tokens where that teacher's specialty matters.

## Why it matters

- **Puts "RL specialists, distill into generalist" on a repeatable footing.** DeepSeek-V3 did this ad-hoc for reasoning only; Motif 3 generalizes to a portfolio of specialties and gives the recipe a name.
- **Avoids the "RL on the frontier model" cost.** Running long-CoT RL directly on a 314B MoE is prohibitively expensive; running RL on smaller domain specialists and distilling is much cheaper.
- **Composes cleanly.** Adding a new specialty means training one new teacher and redoing the distillation stage — not re-running RL on the whole frontier model.
- **Preserves general capability.** Because the distillation is on-policy (from the base student's own rollouts), the student doesn't overfit to any single specialist's style; the KL to a mixture keeps general capabilities from being crowded out by any one specialty.

## Gotchas & tricks

- **Teacher authority routing is load-bearing.** If a math prompt gets equal weight from the safety teacher, the math signal is diluted. Good hard routing on obvious prompts (via a small classifier) matters.
- **Teacher inference cost dominates.** Each rollout token needs one forward per teacher on the same context. Amortize: batch teacher inference, cache prefix computations, and prune inactive teachers per prompt.
- **Specialist drift can compound.** If specialists are trained too aggressively, their per-token distributions become very sharp and the mixture is dominated by the most confident teacher — regardless of actual authority. KL regularization to the shared base during specialist training helps.
- **Order of stages matters.** RL specialists first, then distillation. Distilling before specialists have converged wastes teacher training compute.
- **Doesn't fix conflicting specialists.** If two teachers disagree on a token where both are authoritative (e.g. helpfulness vs safety), the mixture blurs the signal. Motif 3 handles this with careful specialty scoping; general recipes need an explicit conflict-resolution policy.
- **The base's own distribution should be included as a "teacher".** Otherwise the distilled student loses base-model capabilities that no specialist explicitly covered.

## Sources

- Paper: *Motif 3 Technical Report* — Motif Technologies, 2026 — the source paper naming and applying the recipe.
- Related: [on-policy-distillation.md](on-policy-distillation.md) — the underlying OPD family.
- Related: [../case-studies/deepseek-v3.md](../case-studies/deepseek-v3.md) — DeepSeek-V3's ad-hoc single-specialist predecessor.
