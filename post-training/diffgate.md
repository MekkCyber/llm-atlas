# DiffGate — Difficulty-Gated Teacher Guidance

*Depth — adds bounded teacher supervision to GRPO's **all-failure** groups, scaled by group difficulty.*

**TL;DR:** GRPO (and RLVR variants) normalize advantages within a group of `G` rollouts per prompt. If all `G` rollouts fail, the advantages are zero and the policy gets no gradient — a known blind spot on hard problems. DiffGate patches this by injecting *bounded* teacher logprob guidance into those failed trajectories only, scaled by the group's pass rate. Easy prompts get no teacher help; impossible prompts get the most. The GRPO loop is otherwise unchanged.

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md), [fine-tuning/on-policy-distillation.md](./fine-tuning/on-policy-distillation.md)
**Related:** [_rewards.md](./_rewards.md), [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md)

---

## What it is

Standard GRPO group-advantage: `A_i = (r_i − r̄) / σ_r`. If every `r_i = 0`, both `r̄ = 0` and `σ_r = 0` and the group contributes nothing to the gradient. For a model that is at its difficulty frontier, those all-failure groups are exactly the ones you most want to learn from.

DiffGate adds a per-token teacher-logprob loss on the *failed* trajectories, with two gates:

- **Difficulty gate.** Weight the teacher loss by `(1 − pass_rate_group)` — all-success groups get zero teacher pressure; all-failure groups get the most.
- **Bounded clip.** The teacher gradient is clipped so it never dominates the GRPO gradient on groups where both signals disagree.

## How it works

### Per-group terms

For prompt `q` with rollouts `o_1 … o_G` and binary rewards `r_i ∈ {0, 1}`:

```
pass_rate_g  = (1/G) · Σ_i r_i
diff_weight  = 1 − pass_rate_g           ∈ [0, 1]
```

### Teacher loss on failed trajectories only

```
L_teacher = diff_weight · (1/F) · Σ_{i : r_i = 0} Σ_t  −log π_teacher(o_{i,t} | ...) with grad on π_student
```

where `F = |{i : r_i = 0}|` is the number of failed rollouts. In practice this is the standard forward-KL on-policy distillation loss restricted to the failed trajectories.

### Bounded clip

The teacher-loss gradient contribution per token is clipped to a magnitude bound `τ` (hyperparameter). This prevents the teacher-guidance signal from steamrolling GRPO on groups where some rollouts succeed and others don't.

### Combined objective

```
L_DiffGate = L_GRPO + λ · L_teacher
```

`λ` typically 1.0 — the actual balance is controlled by `diff_weight` and the clip.

## Why it matters

- **Covers GRPO's main blind spot.** The all-failure-group pathology is well-known in RLVR practice; DiffGate is a minimal patch that reuses existing infrastructure (you already have the teacher logprobs if you have a teacher).
- **Reported Qwen3 pass@8 gains across all 4 (model-size, domain) settings** in the paper's table — biggest gain on code with the 1.7B student, which is consistent with "harder problems benefit more from teacher help."
- **Composes naturally with other GRPO extensions.** The difficulty-weighting trick is independent of the clip or the ratio surgery; it can combine with [training-inference-mismatch.md](./training-inference-mismatch.md) corrections or scaffolded rationales ([reasoning/rgpo.md](./reasoning/rgpo.md)).

## Gotchas & tricks

- **Needs a teacher.** Just like on-policy distillation, DiffGate requires scoring access to a stronger model. Teacher cost per step is the same as pure OPD on the failed trajectories only, so often cheaper.
- **Teacher-rollout mismatch** can bias the loss if the teacher would never produce the failed student trajectory. Mitigation: use the teacher only in scoring mode on student samples (standard OPD).
- **`τ` matters.** Set it too low and the teacher signal vanishes; too high and successful student trajectories still get pulled toward the teacher. The paper tunes it on a validation set.
- **Pass rate is noisy at small `G`.** With `G = 4`, `pass_rate_g` is on a 5-point grid and `diff_weight` is coarse. Larger `G` makes the gate smoother; typical setups use `G ≥ 8`.
- **Does nothing when the teacher is wrong on hard problems.** If the teacher also fails the same prompts, forward-KL supervision propagates the teacher's errors. Pair with rejection filtering if your verifier is reliable.

## Sources

- Paper: *DiffGate: Difficulty-Gated Teacher Guidance for On-Policy Distillation* — 2026 — [arXiv:2610.04596](https://arxiv.org/abs/2610.04596).
- Related: *GRPO* — Shao et al., DeepSeek, 2024 — see [grpo.md](./grpo.md) for the base algorithm.
- Related: *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* — Agarwal et al., 2024.
