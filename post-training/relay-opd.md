# Relay-OPD — Trajectory-Relayed On-Policy Distillation
*Depth — a bounded teacher handoff that fixes prefix failure in on-policy distillation.*

**TL;DR:** Standard on-policy distillation (OPD) wastes compute when the student picks a bad prefix and every subsequent token is supervised on a doomed continuation. **Relay-OPD** detects those moments via teacher/student *continuation asymmetry* (a label-free trigger), lets the teacher take over for a short leg to redirect the trajectory, then hands control back to the student for training. A capped relay budget concentrates intervention on critical early positions while keeping most of the trajectory on the student's own distribution.

**Prereqs:** [on-policy-distillation](on-policy-distillation.md), [_rl](_rl.md)
**Related:** [_post-training](_post-training.md), [long-cot-rl](reasoning/long-cot-rl.md), [rejection-sampling](rejection-sampling.md)

---

## What it is

A modification to on-policy distillation for reasoning students. Two moving parts:

1. **Handoff trigger.** At each step, compare what the *student* would continue with vs. what the *teacher* would continue with. On a failed prefix, the teacher tends to *redirect* (change reasoning direction) while the student *continues* (compounds the error). The KL / disagreement between the two continuations spikes — this is the trigger.
2. **Relay leg.** When triggered, the teacher generates the next few tokens (bounded by a budget). The student then resumes rollout from that repaired state. All tokens in the relayed trajectory are supervised, but the teacher's leg reshapes the downstream student states so subsequent supervision is informative.

## How it works

```
budget_remaining = B
y = []
for t in 0..T:
    if budget_remaining > 0 and continuation_asymmetry(student, teacher, x, y) > τ:
        # teacher takes the wheel briefly
        for k in 0..L:
            y.append( teacher.sample(x, y) )
            budget_remaining -= 1
    else:
        y.append( student.sample(x, y) )

# now train student on this relayed trajectory y with per-token teacher KL
for t in 0..len(y):
    loss_t = KL( p_T(· | x, y[:t]) || p_S(· | x, y[:t]) )
```

- No labels or verifier needed — the trigger is comparing *policies*, not comparing to ground truth.
- Budget $B$ and leg length $L$ concentrate intervention on early critical positions rather than spreading thin over the whole sequence.

## Why it matters

- Rescues on-policy distillation from its main pathology (prefix failure) without giving up its main strength (training on student-visited states).
- Applies to reasoning models where trajectories are long and one bad token in the first 200 can waste the next 20k.
- Composes with any teacher–student pair; no changes to inference-time behavior.

## Gotchas & tricks

- **Budget too high → drifts back toward off-policy KD.** If the teacher generates most tokens, you lose the on-policy benefit. Empirically small $B$ (concentrated at trigger points) is enough.
- **Budget too low → useless on hard prompts.** Prompts where the student is deeply wrong may need several redirect legs, not one.
- **Trigger threshold τ is sensitive.** Too low → constant teacher takeover, too high → misses real failures. Paper reports robust setting where handoff fires primarily in the first ~10% of positions.
- **Only helps when the teacher is stronger on the disputed state.** If the teacher and student both go wrong, redirect just produces a different wrong trajectory.
- **Log the trigger rate.** Should be low (a few percent of positions). A rising trigger rate mid-training means the student is regressing or the trigger is miscalibrated.

## Reported results

Qwen3-4B-Instruct teacher → Qwen3-0.6B/1.7B students on 8 math reasoning benchmarks:

- **+5.73%** vs. standard OPD on average (1.7B student).
- **+1.49%** vs. strongest baseline FastOPD (1.7B student).
- Consistent improvements at 0.6B.
- **>50% reduction** in trained trajectory length.

## Sources

- Paper: *Pass the Baton: Trajectory-Relayed On-Policy Distillation* — Xu et al., 2026 — [arXiv:2607.26057](https://arxiv.org/abs/2607.26057).
