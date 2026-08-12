# SPOT — Sparse Probing & Outcome-Calibrated Targets
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Fixes two failure modes of vanilla on-policy distillation: reverse-KL over-concentrates on the teacher's mode (missing other plausible continuations) and every position gets the same probing budget. SPOT runs an **acquisition → exploration → exploitation** loop — pick uncertain positions to probe, score teacher-proposed candidates through the *student's own verifier-scored continuations*, then distill toward an outcome-calibrated candidate distribution.

**Prereqs:** [on-policy-distillation.md](on-policy-distillation.md), [grpo.md](grpo.md), [reasoning/prm.md](reasoning/prm.md)
**Related:** [u-opsd.md](u-opsd.md), [rejection-sampling.md](rejection-sampling.md), [rlvr.md](rlvr.md)

---

## What it is

Reverse-KL OPD has two well-known problems: it (a) collapses the student onto the teacher's argmax continuation even when other continuations are also good, and (b) spends probing budget uniformly across all positions, which wastes work on positions where the student is already correct.

SPOT reframes each OPD step as a small **active-learning problem**: decide where in the trajectory to probe (acquisition), decide which teacher candidates to consider (exploration), and decide how much to weight each candidate by its downstream outcome (exploitation). The three sub-decisions combine into a single KL-regularized target distribution that gets distilled into the student.

## How it works

**Acquisition.** For each position `t` in a student rollout, compute a score
```
score_t = f(teacher_entropy_t, top_k_mass_t, student_teacher_mismatch_t)
```
Positions with high teacher entropy (teacher itself uncertain), diffuse top-k mass (many candidates plausible), or large student–teacher divergence get more probing budget. Positions where student and teacher agree with high confidence get skipped.

**Exploration.** At each probed position, sample `m` candidate continuations from the teacher's distribution. For each candidate, generate a student continuation and score it end-to-end with a verifier (rule-based or model-based).

**Exploitation.** Convert verifier scores into a target distribution over candidates: candidates with better downstream outcomes get more mass. Regularize with a KL toward the teacher's original distribution to avoid degenerate targets. Distill the student toward this reshaped target with per-position KL.

**Loop.** Fresh rollouts each iteration.

## Why it matters

- **Budget-efficient OPD.** Probing sparsely at high-value positions cuts the teacher-inference cost of OPD substantially — teacher forward passes are the dominant cost.
- **Outcome-aware target reshaping.** By running each teacher candidate through the student and scoring end-to-end, SPOT rewards continuations that *lead* to good outcomes, not just those with high teacher likelihood at position `t`. This aligns local supervision with global reward.
- **Bridges OPD and RL.** SPOT sits between OPD (per-token teacher KL) and GRPO (per-rollout scalar reward): position-level supervision, outcome-calibrated.
- **Plays with the same stack.** Any GRPO / RLVR / OPD pipeline can swap in SPOT's target constructor without changing rollout or verifier infrastructure.

## Gotchas & tricks

- **Verifier calls compound.** `m` candidates × verified continuations per probed position × probed positions per rollout — the verifier cost multiplies fast. Sparsity of probing (acquisition step) is what keeps this tractable.
- **Acquisition score weights are tuning-sensitive.** Over-weight teacher entropy → probe every uncertain teacher position (many false positives); over-weight mismatch → probe every position the student is already handling differently. Balance empirically.
- **Only makes sense when a verifier exists.** For pure math / code where GRPO already works, SPOT is a competitive alternative; for domains without verifiers, u-OPSD is the closer sibling.
- **The KL anchor prevents reward hacking.** Without the regularization to the teacher's original distribution, the target can collapse onto whatever gets the highest verifier score, even if that's a spurious continuation.

## Sources

- Paper: *Sparse Probing and Outcome Calibration for On-Policy Distillation* — Qu, Zhang, Kong, Shang, Ban, Qiu, Dai, 2026 — the source paper.
- Related: [on-policy-distillation.md](on-policy-distillation.md) for the OPD family SPOT extends.
- Related: [reasoning/prm.md](reasoning/prm.md) — SPOT's position-level scoring is close in spirit to process-reward modeling at a lower abstraction level.
