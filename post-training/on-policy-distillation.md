# On-Policy Distillation
*Depth — distill from a teacher onto trajectories sampled by the student.*

**TL;DR:** Standard knowledge distillation trains a student to match a teacher on the *teacher's* rollouts. **On-policy distillation (OPD)** flips this: the student generates its own trajectory, then per-token loss uses the teacher's distribution at each of those states. This eliminates train/inference mismatch — the student practices on states it will actually visit — but it introduces **prefix failure**: once the student veers into a bad reasoning direction, every subsequent token is trained on a doomed continuation.

**Prereqs:** [_rl](_rl.md), [_post-training](_post-training.md)
**Related:** [rejection-sampling](rejection-sampling.md), [long-cot-rl](reasoning/long-cot-rl.md), [relay-opd](relay-opd.md)

---

## What it is

Given a strong teacher $p_T$ and a weaker student $p_S$, distillation minimizes some divergence between them. Two regimes:

- **Off-policy distillation.** Sample tokens from $p_T$; train $p_S$ to reproduce them (or match $p_T$'s distribution at each token). Simple, but the student is trained on states *the teacher* visits, not states it will visit at inference.
- **On-policy distillation.** Sample tokens from $p_S$ (student rolls out its own trajectory), then take supervision from $p_T$ *at those states*, typically as forward-KL against $p_T$'s next-token distribution.

OPD is closer in spirit to a **soft imitation-learning** algorithm than to classical KD. It resembles PPO with the reward replaced by a token-level KL to the teacher.

## How it works

```
for each prompt x:
    y_S = student.rollout(x)                # states the student actually visits
    for token position t in y_S:
        loss_t = KL( p_T(· | x, y_S[:t]) || p_S(· | x, y_S[:t]) )
    backprop average loss over t
```

- No reward, no verifier. The teacher's next-token distribution *is* the target.
- Because the student rolls out, gradients flow through the exact states the student will produce at inference — no distribution shift.
- Variants scale the loss by teacher confidence, mix off-policy demonstrations, or truncate at the first divergence.

## Why it matters

- **Removes the train/test state gap.** SFT and off-policy KD train on curated / teacher-generated states; the student then sees its own (different) states at inference and drifts. OPD trains directly on the inference distribution.
- **No preference labels, no verifier.** OPD is applicable wherever a stronger teacher exists — general-purpose settings that RLVR can't reach.
- **Smaller than RLHF.** No reward model, no PPO value network, just a teacher forward pass per training token.

## Gotchas & tricks

- **Prefix failure.** Once the student picks a wrong reasoning direction early, all downstream supervision is on a broken trajectory. The teacher-KL at each token is still well-defined, but the tokens the student learns from are conditioned on nonsense. This wastes compute and can *worsen* long-form reasoning. Mitigations: relay-style handoffs ([relay-opd](relay-opd.md)), truncate at first divergence, or filter failed rollouts.
- **Teacher-student continuation asymmetry.** On failed prefixes, the teacher typically *redirects* while the student *continues*. This asymmetry is exploitable — it's a label-free signal that the student has gone off track.
- **Trajectory length grows fast.** Reasoning-model rollouts hit tens of thousands of tokens. Compute scales linearly; a bad prefix wastes most of it. Relay-OPD reports >50% trajectory-length reduction from handoff triggers.
- **Teacher must be stronger *on the student's states*.** A teacher fine-tuned on curated demos may be worse than the student on out-of-distribution student rollouts. Prefer generalist teachers.
- **Not a replacement for SFT.** Bootstrap the student with SFT first so its rollouts are competent enough for teacher supervision to be informative.

## Sources

- Paper: *Pass the Baton: Trajectory-Relayed On-Policy Distillation* — Xu et al., 2026 — names the "prefix failure" mode and the teacher–student continuation asymmetry.
- Related: *Distilling Step-by-Step* and related knowledge-distillation-for-reasoning literature — off-policy precedent.
