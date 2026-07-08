# On-Policy Distillation

*Depth — distill a teacher's per-step behavior into a student while sampling from the student's own policy.*

**TL;DR:** Instead of sampling trajectories from the teacher and imitating them (off-policy distillation) or collecting fresh trajectories for SFT, run the *student*, ask the teacher to score/label the student's states, and update the student on that per-step teacher signal. Keeps the training distribution equal to the deployment distribution while getting the sample-efficiency of imitation. Used by UI-MOPD (2026) to grow a GUI agent across new platforms without forgetting old ones.

**Prereqs:** [_post-training.md](_post-training.md), [_rl.md](_rl.md), [rejection-sampling.md](rejection-sampling.md)
**Related:** [rlvr.md](rlvr.md) · [grpo.md](grpo.md) · [cot-reward-model.md](cot-reward-model.md) · [agents/README.md](../agents/README.md)

---

## What it is

Three data-generation regimes for training a student model from a teacher:

| Regime | Sampled by | Labeled by | Distribution mismatch |
| --- | --- | --- | --- |
| Off-policy distillation | Teacher | Teacher | Student sees teacher-shaped states, not its own |
| SFT on human data | Human | Human | Both fixed; no student in the loop |
| **On-policy distillation** | **Student** | **Teacher** | None — student trains on states it will actually visit |

On-policy distillation combines the distribution-matching property of on-policy RL with the dense supervision of imitation learning. The teacher provides a per-step target (a token distribution, a scalar reward, or a next-action label); the loss is typically a KL from teacher to student.

Contrast with rejection-sampling SFT: rejection sampling filters student rollouts by an outcome verifier and does SFT on survivors. On-policy distillation uses *every* student rollout and gets dense per-step supervision from a teacher, not a binary filter.

## How it works

The core loop is:

```
for step in training:
    prompt / initial state s_0  ~  D_train
    trajectory τ = (s_0, a_0, s_1, a_1, ...)  ~  π_student
    for each (s_t, a_t) in τ:
        target = teacher(s_t)             # distribution, action, or reward
        loss += KL( teacher(s_t) || π_student(· | s_t) )
    update π_student
```

Design choices:

- **Teacher output.** Full token distribution (dense KL); top-k or argmax action (sparser); scalar reward that reshapes the loss (closer to RL).
- **State coverage.** Sample from the student on the current environment/platform; optionally mix in states from prior environments as a distillation-based rehearsal against forgetting.
- **Frozen or moving teacher.** Frozen teacher is safest; a periodically-updated teacher (self-distillation) can help when no stronger teacher exists.

In UI-MOPD's continual GUI-agent setup, each new platform is a new environment. The student samples trajectories there; a stronger teacher labels each state; a mix of prior-platform student-rollouts (still labeled by the teacher) prevents catastrophic forgetting on old platforms.

## Why it matters

- **Distribution match.** The student never sees a state at deployment that it hasn't been trained on — the defining weakness of off-policy imitation.
- **Dense supervision.** Every step contributes; no discarded trajectories the way rejection sampling drops non-passing rollouts.
- **Continual learning without replay buffers.** Old-platform capability is maintained by having the teacher relabel a *few* student rollouts on those platforms, not by storing raw historical trajectories.
- **Composes with RL.** Warm the student with on-policy distillation on states it will visit, then switch to a sparse-reward RL phase (GRPO, RLVR) once the base competence is there.

## Gotchas & tricks

- **Teacher quality caps the student.** If the teacher can't reliably act on the student's states, the KL target is noise. Score the teacher on student rollouts before starting a full run.
- **KL direction matters.** Reverse KL (student toward teacher's mode) can collapse the student; forward KL (teacher's distribution as target) is safer for exploration.
- **Sampling temperature.** Sample the student trajectories at deployment temperature. Sampling too cold collapses state coverage; too hot floods with garbage states that the teacher can't usefully label.
- **Rehearsal ratio.** For continual settings, 5–20% of each batch drawn from prior environments is enough to stop forgetting without dominating the update.
- **Not the same as DAgger.** DAgger is on-policy behavioral cloning with an expert label per state — the SFT variant of this. On-policy distillation typically uses a soft (KL) target rather than a hard action label.

## Sources

- Paper: *UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI Agent Learning* — Chen et al., 2026 — anchoring instantiation for continual GUI agents.
- Paper: *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)* — Ross et al., 2011 — the classical on-policy imitation predecessor.
- Paper: *Distilling the Knowledge in a Neural Network* — Hinton et al., 2015 — the off-policy KL-distillation baseline.
