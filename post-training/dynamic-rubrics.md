# Dynamic Rubrics
*Depth — verifier-free per-step credit assignment for long-horizon agent RL.*

**TL;DR:** For long-horizon agent tasks without a programmatic checker, multi-criteria rubrics scored once per trajectory are a poor per-step signal. DRACO (IBM Research, 2026) generates rubrics **dynamically during training** to track the policy's evolving capability, scores them at trajectory end, then **closed-form redistributes** the scalar judgment over the steps responsible for annotated rubrics — turning it into differentiated per-step advantages for GRPO without training an attribution model.

**Prereqs:** [grpo](grpo.md), [_rewards](_rewards.md), [cot-reward-model](cot-reward-model.md)
**Related:** [rlvr](rlvr.md)

---

## What it is

A reward pipeline for outcome-blind agent training (no ground-truth verifier). Its three pieces plug into GRPO:

1. A **dynamic rubric generator** that emits multi-criteria evaluation rubrics matched to the policy's current capability at each training generation.
2. A **trajectory-level rubric scorer** (an LLM judge) that grades one rubric per criterion at the end of the trajectory and annotates which trajectory step each rubric refers to.
3. A **closed-form advantage redistributor** that turns the (scalar rubric verdict, step annotation) pairs into differentiated per-step advantages consumed by GRPO's policy update.

Distinct from PRM/ORM (which train reward models) and from RLVR (which requires a programmatic verifier).

## How it works

Per training generation:

- The rubric generator writes $k$ criteria appropriate to the policy's current failure modes ("does the shell command respect the requested directory scope?", "does the tool call cite a stub before invoking it?"). "Dynamic" because the rubric set is re-emitted per generation.
- The policy generates trajectories.
- For each trajectory the LLM judge scores each rubric and marks which steps the rubric applies to (via a structured output the judge returns).
- Per-step advantages are computed **closed-form**: each step $t$'s advantage aggregates the contributions of rubrics annotated to $t$, weighted by their verdicts, normalized within the trajectory. No trained attribution model.
- GRPO uses these per-step advantages instead of a broadcast scalar reward.

Because the rubric set evolves, it does not saturate — as the policy solves current criteria, harder ones are introduced.

## Why it matters

- **Verifier-free.** Works where RLVR does not: no programmatic checker required.
- **Per-step credit without a learned attributor.** Closed-form redistribution avoids the compounding-error problem of learned per-step reward models.
- **Beats sparse ground-truth reward in-domain.** On AppWorld, DRACO gains **+15.9 pts over base** and **+5.3 pts over GRPO with sparse ground-truth reward**. On out-of-domain Tau-Bench, **+5.3 pts** over base even without a frontier judge.

## Gotchas & tricks

- **Rubric quality bottlenecks the loop.** If the generator emits rubrics the policy already satisfies trivially, per-step advantage collapses to noise; a difficulty check on new rubrics is essential.
- **Judge cost per trajectory.** Structured per-rubric annotations require capable judges; DRACO shows the effect degrades gracefully with weaker judges out-of-domain but you still want the strongest affordable judge on the training distribution.
- **Complements not replaces RLVR.** For tasks where a verifier does exist, ground-truth reward + DRACO-shaped per-step signal can compose.
- **Watch rubric-to-step attribution reliability.** The judge's step annotation is the load-bearing part of the closed-form redistribution; test that annotations agree with human judgment on a small sample per generation.

## Sources

- Paper: *DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training* — Gandhi et al., IBM Research, 2026 — https://arxiv.org/abs/2609.04094
- Code: https://github.com/IBM/draco
