# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-training method where the **student generates its own rollouts** (on-policy) and is trained to match a stronger teacher's next-token distribution on those rollouts. Occupies a middle ground between SFT (fixed data) and RL (outcome reward): the data distribution tracks the student, but the learning signal is teacher log-probs, not a scalar reward. Increasingly used as a cheaper alternative to RL for reasoning post-training.

**Prereqs:** [_post-training.md](./_post-training.md), basic SFT.
**Related:** [grpo.md](./grpo.md) · [rejection-sampling.md](./rejection-sampling.md) · [delta-distillation.md](./delta-distillation.md) · [dpo.md](./dpo.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md)

---

## What it is

**SFT** trains on a fixed corpus — data distribution ≠ student's own. **RL** samples from the student but supervises with a scalar reward (high variance, sparse). **On-policy distillation** samples from the student *and* supervises with a full next-token distribution from a teacher:

$$
L_{\text{OPD}} = \mathbb{E}_{y\sim \pi_S(\cdot\mid x)}\bigl[\; \mathrm{KL}\!\bigl(\pi_T(\cdot\mid x,y_{<t}) \,\|\, \pi_S(\cdot\mid x,y_{<t})\bigr)\;\bigr]
$$

Dense per-token signal (like SFT), on-policy data distribution (like RL), no reward function needed.

## How it works

**Loop.** Sample $y \sim \pi_S(\cdot\mid x)$. Forward the teacher $\pi_T$ on $x, y$. Compute per-token KL (or cross-entropy against the teacher's top-K). Backprop into $\pi_S$.

**Why on-policy matters.** Off-policy KD (train student to match teacher on the teacher's rollouts) suffers from exposure bias — at inference the student sees its own prefixes, which look nothing like the teacher's. On-policy KD matches the deployment distribution.

**Teacher choice.** Any stronger model works: a bigger open model, a post-trained sibling, or the same model with privileged inputs. When the teacher is the student's *own post-trained future* (say, an RL checkpoint), OPD becomes a distillation of RL gains into a cheaper student — the setting where OPD is currently most useful.

**Cost.** One student forward + one teacher forward per token. Teacher can be quantized; if the teacher is much larger, that dominates. Still typically cheaper than GRPO because there's no reward-model inference and no G-way group sampling.

## Why it matters

- **Cheaper than RL for reasoning post-training** in the regime where a strong teacher exists — you skip the reward model, the multi-rollout groups, and the value-network debates.
- **Dense supervision.** Unlike RL's scalar reward, every token gets a signal — much lower variance, faster convergence.
- **Composes with RL.** Common recipe: RL a large teacher, then OPD-distill it into a smaller deployable student.
- **Reasoning-friendly.** Works well for math, code, and CoT-style outputs where a strong post-trained teacher exists but running RL on the student is too expensive.

## Gotchas & tricks

- **KL is unbounded when the student assigns near-zero mass to a token the teacher wants.** In practice, use top-K teacher distributions (K=20–50) or a mixed CE + KL objective.
- **Teacher must be stronger *at the deployment distribution*.** A generalist teacher can be worse than a specialist student on a narrow task; check with a small SFT run first.
- **Exposure to garbage.** Early in training, student rollouts include hallucinated garbage. The teacher's supervision on garbage prefixes may be low-quality; short warmup on curated data helps.
- **Delta distillation** ([delta-distillation.md](./delta-distillation.md)) fixes a subtle problem with vanilla OPD: matching the teacher's full distribution copies the teacher's base-model habits too, not just its post-training gains.
- **Language drift.** For multilingual reasoning, distilling only from an English teacher shifts responses toward English — mix in target-language rollouts.

## Sources

- Paper: *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* — Agarwal et al., Google DeepMind, 2023 — the original OPD framing.
- Paper: *On-Policy Delta Distillation for Multilingual Math Reasoning* — 2026 — [arXiv:2608.05802](https://arxiv.org/abs/2608.05802) — extends OPD to multilingual reasoning; motivates [delta-distillation.md](./delta-distillation.md).
