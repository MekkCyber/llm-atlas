# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Sample rollouts from the student policy itself, then score each token with a teacher's per-token distribution and distill *that* signal back into the student. Unlike offline distillation, the training distribution stays on-policy — the student improves along paths it actually visits — which is why OPD sits so naturally next to GRPO-style outcome RL. Recent work shows OPD is best understood as an **exploration catalyst** (it doesn't raise the capability ceiling; it steers the student toward correct paths inside it), but the naive form has two failure modes that need lightweight regularization.

**Prereqs:** [_rl.md](./_rl.md), [grpo.md](./grpo.md)
**Related:** [rlvr.md](./rlvr.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md) · [reasoning/length-penalty.md](./reasoning/length-penalty.md)

---

## What it is

Standard *offline* distillation trains a student to match a teacher on a fixed corpus of teacher outputs. **On-policy distillation** flips the sampling: the student rolls out its own trajectories, and the teacher provides *token-level* guidance on those rollouts. The training distribution is always the student's current distribution, so the student never wastes updates on trajectories it would never sample.

Compared to offline distillation, OPD:

- Sees only trajectories in the student's *support* (no distribution shift at deployment).
- Provides dense per-token supervision that outcome-only RL lacks.
- Combines naturally with an outcome-reward loss (GRPO, RLVR) — you sum a token-level distillation term with a trajectory-level advantage term.

## How it works

1. Student policy $\pi_\theta$ samples a rollout $o = (o_1, \ldots, o_T)$ from prompt $q$.
2. Teacher $\pi_T$ (frozen, usually larger, or the student itself with a skill-augmented context) is evaluated on the same rollout to produce per-token distributions $\pi_T(\cdot \mid q, o_{<t})$.
3. Per-token distillation loss (forward KL, reverse KL, or cross-entropy over the top-k) provides a **dense** gradient signal along the rollout.
4. This loss is optimized *jointly* with an outcome-based RL loss over the same rollouts. The outcome term (GRPO / RLVR) tells the student which trajectories were right; the distillation term tells it which *tokens within each trajectory* to prefer.

Two shapes of OPD in recent work:

- **External-teacher OPD** — a larger frozen model provides the token distribution. Classic teacher-student distillation, sampled on-policy.
- **Self-distillation OPD** — the *student* provides both distributions, but the teacher context is augmented with a hindsight description of what the trajectory achieved (Seed, 2026). Removes teacher-cost, generates the signal purely from behavioral self-supervision.

## Why it matters

- Fills the token-level gradient hole in RL post-training. Trajectory-level advantages (GRPO) give one scalar per rollout; OPD gives $|o|$ signals per rollout.
- Doesn't require a preference reward model — cheap teacher access is enough.
- Explains a chunk of the gap between "raw GRPO" and "GRPO + strong distillation warm-start" seen in R1-style pipelines.
- Recent analysis (2026) reframes OPD's *purpose*: it steers exploration inside the student's existing capability set, not beyond it. Practical consequence — pick teachers whose *reasoning shape* fits the student, not just whichever is biggest.

## Gotchas & tricks

- **Student-Teacher Mismatch.** If the teacher-student distributional gap is too large, the guiding signal aims at paths the student can't actually walk, and the RL update points in a counterproductive direction. Cap the teacher size relative to the student, or clip the advantage from the distillation term.
- **Length Exploitation.** Aggregating a per-token distillation loss creates length-dependent shortcuts — the student can lower average loss by truncating or padding rather than reasoning better. Two lightweight regulators from Demystifying-OPD (2026): **advantage clipping** on the per-token distillation contribution, and **log-scale compression** of the guiding signal so long rollouts don't dominate.
- **Signal quality > teacher scale.** A well-regulated small teacher beats an unregulated bigger one. When OPD doesn't help, look at the signal before you look at the teacher.
- **Doesn't raise the ceiling.** OPD steers the student inside its capability set, not outside. Pair with SFT / mid-training if the student can't reach the answer at all.
- **Interacts with KL to reference.** The reference-KL from the outcome-RL loss and the token-distillation KL to the teacher can pull the student in different directions. In practice, weight the outcome-KL smaller when OPD is on; the distillation term already provides a stable anchor.

## Sources

- Paper: *Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations* — 2026 — the diagnostics and the two regulators.
- Paper: *Seed: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning* — Wu et al., 2026 — self-distillation via hindsight-skill contexts, joint with GRPO.
- See also: [grpo.md](./grpo.md), [rlvr.md](./rlvr.md) for the outcome-RL objectives OPD stacks with.
