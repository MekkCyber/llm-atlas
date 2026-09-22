# On-policy distillation (OPD) & calibration
*Depth — distilling a teacher into a student via the teacher–student discrepancy on the student's own rollouts, and correcting for teacher-side noise.*

**TL;DR:** On-policy distillation trains a student policy by (1) sampling a rollout from the *student* itself and (2) using the *teacher's* next-token distribution on that rollout as the learning target — a per-token KL between student and teacher, computed on the student's on-policy trajectories. This is standard-issue for compressing reasoning models. But the raw teacher–student discrepancy also contains *teacher-side noise*: the teacher's own idiosyncratic likelihood shifts, unrelated to the capability gap. **Cal-OPD** (He et al., 2026) estimates that noise region via positive/negative privileged interventions and calibrates the target to keep only the discrepancy beyond it — outperforming vanilla and privileged OPD on math reasoning while using only 52–65% of the raw signal.

**Prereqs:** [_rl.md](./_rl.md) (on-policy sampling, KL loss), [grpo.md](./grpo.md) (student rollout mechanics)
**Related:** [long-cot-rl.md](./reasoning/long-cot-rl.md) · [long2short.md](./reasoning/long2short.md) · [rejection-sampling.md](./rejection-sampling.md)

---

## What it is

A distillation regime where the *student* generates the trajectories used for training and the *teacher* provides token-level next-token distributions as targets. Distinct from off-policy distillation (SFT on teacher rollouts) because the student sees its *own* mistakes, and distinct from RL because the reward is soft (KL to the teacher's distribution), not scalar.

Cal-OPD is a *calibration* on top of OPD: it observes that the teacher–student discrepancy has two components — real capability gap + teacher-specific noise — and subtracts an estimate of the second.

## How it works

**Vanilla OPD.** For each token position $t$ in a student rollout $o$:

$$
L_{\text{OPD}}(t) = \mathrm{KL}\!\left(p_S(\,\cdot\, \mid o_{<t}) \,\|\, p_T(\,\cdot\, \mid o_{<t})\right)
$$

Sum over tokens, backprop into the student.

**Privileged OPD.** Give the teacher extra context (a reference answer, a hint, a longer prompt) not shown to the student. The teacher's distribution is now sharper and the student learns from a more capable target. But the added conditioning *also shifts the teacher's own likelihoods in ways unrelated to student ability* — this is the "teacher-side noise" Cal-OPD targets.

**Cal-OPD.** For each token, estimate the teacher's *self-deviation region* by comparing its distribution under paired interventions:

- Positive intervention: teacher sees privileged information.
- Negative intervention: teacher sees a control-matched but uninformative version.

The **spread** of the teacher's distribution across these two conditions defines the teacher-noise region. The calibrated learning signal is the projection of the student–teacher discrepancy *onto the complement* of that region — keeping only the component that lies beyond the teacher's own drift.

Empirically, this retains 52–65% of the raw discrepancy magnitude, and drives strictly better math-benchmark performance than raw OPD across scales.

## Why it matters

- **Cleaner distillation signal.** OPD has become one of the standard tools for compact reasoners; anything that improves its signal-to-noise ratio propagates broadly.
- **Explains privileged OPD's mixed results.** Privileged-info OPD was already known to be sensitive to *what* you privilege the teacher with. Cal-OPD's diagnosis — larger privileged shifts also inject more teacher noise — is a satisfying account of why.
- **Causal-intervention framing.** Diagnosing training-signal contamination by contrasting matched positive/negative conditions is reusable: the same shape of intervention could be applied to reward-model training or RL reward shaping.

## Gotchas & tricks

- **You need a controllable teacher.** Estimating the self-deviation region requires running the teacher on the paired positive/negative interventions. That's cheap if the teacher is local, expensive if it's an API.
- **Which axis of the distribution counts as "noise" matters.** The paper projects the discrepancy onto the complement of the teacher-noise region, but there are multiple reasonable definitions of "outside." Ablations in the paper settle on a specific one; naive implementations may not reproduce.
- **Not a rescue for a weak teacher.** Cal-OPD only helps if there's real signal beyond the teacher's noise. If the teacher is barely better than the student, calibrating away 40% of the signal leaves you with nothing.
- **Composable with rejection sampling.** Filtering student rollouts before applying OPD (rejection-sampling style) reduces variance orthogonally. The two combine naturally.

## Sources

- Paper: *Calibrating Teacher–Student Discrepancy for On-Policy Distillation* — He, Li, Chen, 2026 — [arXiv:2609.21619](https://arxiv.org/abs/2609.21619).
- Related: privileged OPD variants, standard SFT-on-teacher-rollouts distillation.
