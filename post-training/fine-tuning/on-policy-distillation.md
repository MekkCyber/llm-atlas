# On-Policy Distillation (OPD)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A distillation regime in which the *student* generates rollouts and the *teacher* provides per-token (or per-prefix) supervision on those student rollouts — instead of distilling from teacher-generated rollouts (offline distillation). OPD removes the prefix-mismatch problem of offline distillation: the student is supervised exactly on the distribution it would produce at inference. The cost is fragility at the start of training, when raw student rollouts are too poor for teacher supervision to do useful work.

**Prereqs:** [../_post-training](../_post-training.md), [../_rl](../_rl.md)
**Related:** [trust-region-behavior-blending](trust-region-behavior-blending.md) · [../grpo](../grpo.md) · [../rejection-sampling](../rejection-sampling.md)

---

## What it is

Classical distillation generates training data by sampling from the *teacher* — the student is trained on prompts and teacher completions. The student learns to imitate the teacher pointwise on those completions, but its own rollouts at inference may visit prefixes the teacher never produced. The result is a distribution mismatch: the student's quality degrades as it gets further into a rollout from any state the teacher would have avoided.

On-policy distillation flips the sampling. The *student* generates the rollout. The teacher is consulted on the student's rollout — typically producing a target distribution at each token (or scoring each prefix). The loss is a divergence (commonly reverse KL or a related discrepancy) between student and teacher distributions on the student's own trajectories.

The student is now optimized exactly for the distribution it actually rolls out at inference. The mismatch closes.

---

## How it works

### The loop

```
for each step:
    1. Sample a rollout o_1:T ~ π_student(· | q)            # student is on-policy
    2. For each prefix (q, o_<t), query teacher
         p_teacher = π_teacher(· | q, o_<t)
       and compute student
         p_student = π_student(· | q, o_<t)
    3. Loss = Σ_t KL( p_student || p_teacher )              # reverse KL, typically
    4. Backprop into student
```

The teacher is frozen; the student updates. Compared to RL, there is no reward function — the supervision is *what the teacher would have done*. Compared to offline distillation, there is no static dataset — the student's policy continuously shapes what gets supervised.

### Reverse KL (the OPD default)

Most OPD recipes use reverse KL ($\mathrm{KL}(\pi_\text{student} \| \pi_\text{teacher})$). Reverse KL is *mode-seeking*: the student picks one consistent way to behave and matches the teacher tightly there, rather than spreading mass to cover all teacher modes. For distilling a strong reasoner into a smaller model, mode-seeking is usually the right inductive bias.

### Why early rollouts are a problem

A randomly-initialized or under-trained student visits prefixes (incoherent text, off-task continuations) that the teacher itself would never produce. The teacher's supervision on those prefixes is degenerate: the teacher tries to recover, but the student gets weak signal on prefixes far outside any sane policy's support. The student wastes optimization on garbage prefixes early in training. This is the fragility OPD is famous for, and motivates warmup techniques like [trust-region-behavior-blending](trust-region-behavior-blending.md).

---

## Why it matters

- **No prefix mismatch.** The student is supervised on its own distribution, so train-test gap collapses. This is the structural argument for OPD over offline distillation.
- **No reward function needed.** Cheaper than RL: skip reward design, skip critic training, skip credit assignment.
- **Cheaper than rolling out the teacher.** The teacher is queried only on student-generated prefixes (which the student also generates), not used as a sampling oracle — fewer big-model forward passes per training step than offline distillation with teacher rollouts.
- **The default for small-model reasoners in 2025+.** R1-style results showed that running RL on small models is wasteful when a strong reasoning teacher exists; OPD-from-traces is the modern recipe.

---

## Gotchas & tricks

- **Cold start.** Raw student rollouts are bad early on. Warmup with a behavior policy closer to the teacher (e.g. [trust-region-behavior-blending](trust-region-behavior-blending.md)) or with a small phase of offline distillation, then switch to pure OPD.
- **Reverse vs forward KL.** Reverse KL is mode-seeking (sharp student, may miss teacher diversity). Forward KL is mass-covering (broader student, may include teacher modes the student can't actually realize). Default to reverse KL for capability transfer; forward KL for stylistic averaging.
- **Teacher freshness.** A teacher much stronger than the student gives crisp supervision but also large gradients. A near-peer teacher gives subtler signal — sometimes preferable for stability.
- **Per-token vs per-prefix loss.** Per-token KL is the standard; some variants compute a single scalar score per rollout (closer to RL with a teacher-as-reward). Per-token is cheaper to supervise but assumes the teacher's per-step distributions are calibrated.
- **Not RL.** OPD has no exploration term. It only refines toward the teacher's behavior, so it cannot exceed the teacher. If you want capability beyond the teacher, use RL with a reward signal, not OPD.

---

## Sources

- Paper: *Trust-Region Behavior Blending for On-Policy Distillation* — Plyusov, Gorbatovski, Malakhov, Balagansky, Shaposhnikov, Korotyshova, Gavrilov, 2026 — T-Tech. Uses the OPD framework and proposes a trust-region warmup; see [trust-region-behavior-blending](trust-region-behavior-blending.md).
- Background: *DeepSeek-R1* — DeepSeek, 2025 — small-model R1 distillation shows OPD-style trace-distillation beats RL-on-small-models.
