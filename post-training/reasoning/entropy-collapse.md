# Entropy Collapse (at the Entrance)
*Depth — the RLVR-induced diversity loss localized to the first tokens of a reasoning trajectory.*

**TL;DR:** RLVR raises pass@1 but shrinks the reachable solution space, hurting test-time-scaling. Li (Birmingham, 2026) shows on Countdown that the loss is **concentrated at the entrance** — the first operand + operator — where per-token likelihood shifts are 11–16× larger than in downstream reasoning. Solutions remain executable; the policy just stops initiating them. Late-layer parameter interpolation with early checkpoints restores +37% coverage at no loss in pass@1; staged SFT → DPO → RLVR pipelines preserve early-step entropy in the first place.

**Prereqs:** [rlvr](../rlvr.md), [grpo](../grpo.md), [ppo](../ppo.md)
**Related:** [dpo](../dpo.md), [long-cot-rl](long-cot-rl.md), [model-souping](../../pre-training/model-souping.md)

---

## What it is

A measurable failure mode of RLVR: the fraction of enumerable *solution families* the policy can produce for a task falls sharply after RL, even while pass@1 rises. On Countdown (each puzzle has a discrete set of solution families defined by the first operand and operator), coverage drops by up to **67%**, and cuts in half even on puzzles the policy solves at every checkpoint.

The paper splits the failure into two possibilities and empirically settles it:

- **Access failure** — the policy cannot produce the first token of a family's solution.
- **Execution failure** — the policy can produce the first token but cannot complete the computation.

Access dominates. The door is locked; the room inside is fine.

## How it works

Two constructive interventions follow from the localization:

1. **Late-layer parameter interpolation with earlier checkpoints.** Interpolate the late layers only of the RLVR checkpoint with the same layers from an early (higher-entropy) checkpoint. Recovers **+37% solution coverage** at **no loss in pass@1**. Early layers (feature extraction) stay RL-tuned; late layers (decoding) inherit some of the earlier distribution's breadth.
2. **Staged post-training.** SFT alone preserves >2× the coverage of pure RLVR. Staging as **SFT → DPO → RLVR** preserves early-step entropy through the pipeline — DPO's KL-tethering to the SFT reference model is doing prevention work RLVR alone lacks.

Surface prompting fixes ("think about a different approach") do not recover diversity; this is a *policy* problem, not a decoding-strategy problem.

## Why it matters

- **Names the mechanism** behind the well-known pass@k collapse after RL post-training.
- **Recoverable with cheap surgery.** Late-layer interpolation is a post-hoc merge — no re-training required. Useful when only the final RLVR checkpoint is available.
- **Prescription for training pipelines.** Staged SFT → DPO → RLVR is a straightforward change that preserves breadth prophylactically.

## Gotchas & tricks

- **Entropy collapse recurs across benchmarks, not just Countdown.** The paper confirms it on six math benchmarks with 7B and 14B models — but Countdown is the only setting where solution families are cleanly enumerable, making it the right diagnostic testbed.
- **Not inevitable.** Early-step entropy collapse is a bias of naive RLVR, not an unavoidable byproduct of reasoning optimization; the DPO stage is what breaks the pattern.
- **Interpolation ratio matters.** The paper reports one recipe; sweep the interpolation coefficient — coverage and pass@1 trade off around a small window.
- **Distinct from length collapse.** This is about *starting* diversity, not about response-length compression. See [length-penalty](length-penalty.md) for the length-side story.
- **Test with pass@k, not pass@1.** RLVR gains are hidden by pass@1; you only see the collapse in pass@k or in explicit family-coverage measurements.

## Sources

- Paper: *Locked at the Entrance, Open Inside: Where RLVR Narrows the Solution Space* — Ruizhe Li, University of Birmingham, 2026 — https://arxiv.org/abs/2608.29188
- Paper: *Kimi k1.5* — Moonshot AI, 2025 — earlier evidence of pass@k collapse under RL post-training.
