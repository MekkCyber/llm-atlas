# On-Policy Distillation

*Depth — distill a stronger teacher into a student using the student's own rollouts as the sample distribution; the "delta" variant learns from the teacher's post-training gain rather than its raw distribution.*

**TL;DR:** On-Policy Distillation (OPD) is a distillation objective that mirrors an RL setup: the student generates rollouts, and each token is trained toward the teacher's log-probability on the student's own outputs. It's pitched as an RL alternative for post-training — same on-policy sample structure as GRPO but with a distillation loss instead of a policy-gradient loss. **On-Policy Delta Distillation (OPD²)** improves on this by using the log-prob *gap between the post-trained teacher and its base model* — the teacher's post-training "delta" — instead of the teacher's raw log-prob, isolating what post-training added.

**Prereqs:** [rlvr.md](rlvr.md), [rejection-sampling.md](rejection-sampling.md)
**Related:** [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [reasoning/long2short.md](reasoning/long2short.md), [dpo.md](dpo.md)

---

## What it is

Three roles in the recipe:

| Role | Model | Purpose |
| --- | --- | --- |
| **Student** | The model being trained | Generates the rollouts; receives the distillation gradient |
| **Teacher** | A post-trained stronger model (usually a fine-tuned or RL'd sibling) | Provides the target log-probabilities |
| **Teacher base** (OPD² only) | The teacher's *pre-post-training* checkpoint | Subtracted from the teacher's log-probs to isolate the post-training delta |

OPD's structural advantage over classical distillation is the *on-policy* sample distribution: the teacher scores tokens the *student* actually produces, so training signal targets the regions of behavior the student would generate at inference. Classical distillation runs teacher-generated data through the student and can leave the student weak in its own generation regime.

---

## How it works

### OPD (baseline)

1. **Rollout.** Student samples response `y` for prompt `x`: `y ~ π_student(·|x)`.
2. **Score with teacher.** For each token `y_k`, compute `log π_teacher(y_k | x, y_<k)`.
3. **Loss.** Minimize `−E[log π_teacher(y_k|·)]` under the student. In practice, add a KL term (`KL(π_student ‖ π_teacher)` on the student's tokens) so the student's distribution shape moves toward the teacher — not just its argmax.
4. **Optimizer.** Standard on-policy RL loop (rollout → score → update → new rollout), replacing the PPO/GRPO advantage with the teacher log-prob.

### OPD² (delta variant)

Replace the target `log π_teacher(y_k|·)` with the **delta**:

```
Δ_k = log π_teacher(y_k|·) − log π_teacher_base(y_k|·)
```

where `π_teacher_base` is the teacher *before* its own post-training (e.g., Qwen3-Base for a Qwen3-post-trained teacher). The student is trained to increase log-prob on tokens where the teacher's post-training amplified them, and decrease log-prob where post-training suppressed them.

The delta isolates *what post-training added* to the teacher, and prevents the student from being punished for token preferences it already shares with the base model.

## Why it matters

- **RL alternative for post-training.** Distillation is stable, cheap per step, and doesn't need a reward function or a verifier — a real advantage where verifiers don't exist.
- **Delta filters out shared priors.** OPD²'s subtraction is a general lever: any two models sharing a base can distill only the delta between them, avoiding regression toward the base's preferences.
- **Multilingual generalization.** On multilingual math (English / Korean / Japanese, Qwen3), OPD² beats OPD, with the largest gains in Korean and Japanese; also narrows the English–Korean gap.
- **Language-preservation lever.** English-only distillation can leak into Korean/Japanese performance but shifts responses *toward English*. The delta variant plus target-language data preserves language identity.

## Gotchas & tricks

- **On-policy is expensive.** Rollouts per update step multiply generation cost; typical is smaller mini-batches than classical distillation, more of them.
- **Teacher-base availability.** OPD² needs the teacher's *un-post-trained* checkpoint. If the teacher was released without its base, you can't do OPD² cleanly — an approximate base (a smaller foundation model) is a workable but noisy substitute.
- **Language drift.** A monolingual teacher will pull the student toward that language even for multilingual prompts. Add target-language rollouts or a language-consistency term.
- **Distinct from log-prob KD.** Classical KD scores *teacher-generated* tokens. OPD scores *student-generated* tokens. This distinction dominates the technique's behavior in practice.
- **Not a replacement for RL where reward hacking matters.** OPD inherits whatever biases the teacher has. If the teacher was already reward-hacked, the student will be too. RLVR-then-distill is safer than distill-only.

## Sources

- Paper: *On-Policy Delta Distillation for Multilingual Math Reasoning* — Anonymous, 2026 — [arXiv:2608.05802](https://arxiv.org/abs/2608.05802). Introduces OPD² and evaluates on English / Korean / Japanese math benchmarks with Qwen3.
- Related: on-policy distillation as an RL alternative connects to the broader post-training landscape ([rlvr.md](rlvr.md), [reasoning/long2short.md](reasoning/long2short.md)).
