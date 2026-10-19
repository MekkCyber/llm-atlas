# On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Classical distillation feeds a student its *own teacher's* samples and matches the two output distributions. **On-policy distillation (OPD)** flips the sampling: the *student* generates the responses, and the teacher scores them. The gradient becomes a token-level KL between teacher and student on the student's own trajectories — closer in spirit to policy-gradient RL than to supervised distillation. It is the standard way to transfer a stronger teacher's capability into a smaller or compute-constrained student without the exposure bias of teacher-sampled data.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md), [rejection-sampling.md](./rejection-sampling.md)
**Related:** [rlvr.md](./rlvr.md), [_post-training.md](./_post-training.md)

---

## What it is

Given a teacher $\pi_T$ (strong, frozen) and a student $\pi_\theta$ (being trained), classical knowledge distillation minimizes

$$
\mathcal{L}_{\text{KD}} = \mathbb{E}_{o \sim \pi_T(\cdot|q)} \big[\, \mathrm{KL}(\pi_T(\cdot|q, o_{<t}) \,\|\, \pi_\theta(\cdot|q, o_{<t})) \,\big]
$$

i.e. we score teacher-generated completions. **On-policy distillation** samples from the student instead:

$$
\mathcal{L}_{\text{OPD}} = \mathbb{E}_{o \sim \pi_\theta(\cdot|q)} \big[\, \mathrm{KL}(\pi_T(\cdot|q, o_{<t}) \,\|\, \pi_\theta(\cdot|q, o_{<t})) \,\big]
$$

The teacher is used only as a *scoring oracle* over the student's trajectories — never as a sampler.

---

## How it works

1. For each prompt $q$ in the batch, sample $G$ completions $o_i$ from the student $\pi_\theta$ (same rollout machinery as GRPO).
2. For every token position of every student completion, run the frozen teacher $\pi_T$ to get its next-token distribution.
3. Compute per-token KL between teacher and student. The loss is the mean KL over all student tokens.
4. Standard policy-grad tricks apply: PPO-style ratio clipping on the student's logprobs, group-mean baseline if KL is treated as a reward, KL-to-reference anchor against the SFT base.

The resulting update is: *"On tokens where I sampled, pull my distribution toward the teacher's distribution."* Because tokens come from the student, the training distribution matches what the student will see at inference — no exposure bias.

---

## Why it matters

- **No teacher sampling cost.** Teacher only runs forward passes (scoring), never generation. For an $N\times$ larger teacher this is roughly a $K\times$ saving where $K$ is the completion length.
- **Transfers RL-shaped policies cheaply.** If the teacher got good via expensive RL, OPD lets much smaller students inherit the gains with a single scoring-only pass — DeepSeek-R1-Distill, Qwen2.5-R1-Distill, and most "distilled reasoner" families use variants of this loop.
- **Reduces exposure bias.** The student is trained on text it would actually produce, not idealized teacher text. This is the same reason RLVR beats pure SFT: the training distribution matches the inference distribution.

---

## Gotchas & tricks

- **Rollout quality is a floor.** If the student is too weak to ever land near a correct trajectory, teacher scoring gives little signal — some papers mix a small fraction of teacher-sampled completions early in training.
- **Teacher-student vocab must match.** Scoring assumes a shared tokenizer. If vocabularies differ, you need either a projection or forward-KL on an aligned vocab subset.
- **Blends naturally with RL.** OPD + verifiable reward on the same rollout batch is one of the cheapest recipes for "distill into a specific size, then polish with RLVR."
- **Direction of KL matters.** Forward KL ($\pi_T \,\|\, \pi_\theta$) is mode-covering; reverse KL ($\pi_\theta \,\|\, \pi_T$) is mode-seeking. OPD typically uses forward KL (matches the teacher's full support on student tokens).

---

## Sources

- Paper: *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* — Agarwal et al., 2023 — formalizes the student-samples / teacher-scores loop.
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — the R1-Distill family uses on-policy-flavored distillation to transfer R1's reasoning into Qwen/Llama students.
- Paper: *Waypoint-OPD* — Jiang et al., 2026 (arXiv 2610.21601) — gated extension that uses the pre-vs-post-RL *shift* in teacher logits as the training signal; see [waypoint-opd](./waypoint-opd.md).
