# SAF-OPD — Stable Advantage Fusion for On-Policy Distillation
*Depth — stabilizing the fusion of GRPO's response-level advantage with OPD's token-level advantage.*

**TL;DR:** Combining GRPO (verifiable, response-level reward) with on-policy distillation (dense, token-level teacher signal) is attractive but fragile: a fixed-coefficient fusion causes **entropy collapse** because of two miscalibrations — magnitude mismatch and temporal mismatch. SAF applies two lightweight, independently-toggleable pipelines *only* to the OPD advantage — **sparsify-then-compress** for magnitude, **warm-up-then-anneal** for time — and preserves both the exploration RLVR provides and the density OPD provides.

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md), [w2s-opd.md](./w2s-opd.md)
**Related:** [long-cot-rl.md](./reasoning/long-cot-rl.md), [cripo.md](./cripo.md)

---

## What it is

A pipeline that lets GRPO and on-policy distillation train the same policy without one signal drowning out the other. Diagnoses the standard "GRPO + OPD, fixed λ" recipe as broken because:

- **Magnitude mismatch:** token-level OPD advantages spike orders of magnitude above the bounded RLVR advantage, erasing the RLVR signal.
- **Temporal mismatch:** full-strength OPD throughout training keeps pulling the student toward the teacher and blocks the exploration RLVR needs to surpass it.

## How it works

Applied only to the OPD advantage `A_OPD` (leaves `A_RLVR` untouched):

1. **Sparsify.** Drop tokens whose `|A_OPD|` is below a per-batch percentile — usually the noisy majority.
2. **Compress.** Clip the surviving `A_OPD` values to a percentile-based cap so no single token dominates.
3. **Warm-up.** During the first phase of training, ramp the OPD contribution from 0 to full to give the policy time to establish RLVR-driven behavior.
4. **Anneal.** In the final phase, decay OPD back down so exploration takes over and can surpass the teacher.

Each stage is independently switchable and adds negligible per-token overhead. Fusion combines the two advantages as `A_total = A_RLVR + β(t) · A_OPD` with `β(t)` the warm-up/anneal schedule.

## Why it matters

- Turns "GRPO + OPD" from a promising-but-collapsing combination into a reliable recipe: +0.51–2.70% aggregate score across Qwen3-{1.7B, 4B, 8B} × math + code benchmarks, without entropy collapse.
- Generalizes: the magnitude/temporal diagnostic is applicable to any hybrid RL-KD setup where signals of very different densities meet.

## Gotchas & tricks

- Sparsify **before** compress; compressing an unsparsified advantage still lets the noisy majority dominate.
- The four stages compose but are not all-or-nothing — magnitude control is typically the higher-leverage lever; temporal control matters more at longer training.
- Watch entropy (not loss) as the collapse indicator; loss can look healthy while the policy is already saturating.

## Sources

- Paper: *SAF-OPD: Stable Advantage Fusion for On-Policy Distillation* — Ding et al., 2026 — [arXiv:2607.29209](https://arxiv.org/abs/2607.29209)
