# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** OPD trains a smaller *student* model on the outputs it produces *itself*, correcting each next-token prediction toward the *teacher's* distribution on that same context. Unlike sequence-level knowledge distillation (SeqKD), the training distribution is the student's own rollout, not a fixed teacher trace — closing the exposure-bias gap between distillation-time and inference-time contexts.

**Prereqs:** [_post-training](./_post-training.md), [rejection-sampling](./rejection-sampling.md)
**Related:** [rlvr](./rlvr.md), [grpo](./grpo.md)

---

## What it is

Classical knowledge distillation gives the student teacher-generated tokens and asks it to imitate them. Two problems: (1) the student never sees its own errors, so at inference it walks off the training distribution; (2) it never learns the teacher's full distribution, only the greedy trace.

On-policy distillation fixes both:

1. **Roll out with the student.** For each prompt, generate a full response using the *student's* current policy.
2. **Score with the teacher.** For each context $(q, o_{<t})$ the student produced, evaluate the *teacher's* next-token distribution $p_T(\cdot \mid q, o_{<t})$.
3. **Match the student to the teacher.** Minimize a per-token divergence — usually forward KL — between $p_S$ and $p_T$ at each of the student's own contexts.

Because the student trains on the contexts it will actually see at inference, the exposure-bias gap between distillation and generation closes.

## How it works

Let $\tau = (q, o_1, ..., o_T)$ be a student rollout. The OPD objective at step $t$ is:

$$
\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{\tau \sim \pi_S} \sum_t \mathrm{KL}\!\left(p_T(\cdot \mid q, o_{<t}) \,\|\, p_S(\cdot \mid q, o_{<t}; \theta)\right)
$$

Practical variants:

- **Token-level KL** on the LM-head softmax (standard). Cheap; only needs one teacher forward pass per student token.
- **Latent-state alignment** — additionally align intermediate hidden states between student and teacher (OPRD, LastOPD). Motivated by wanting to teach *how* the teacher thinks, not just *what* it says. Prone to collapse when depth-paired layers have different roles across model sizes — see [lastopd-latent-supervision](./lastopd-latent-supervision.md) (not yet written) for the observed failure mode and a fix.
- **Multi-teacher OPD (MOPD)** — mix per-token supervision from several specialist teachers. Token-level routing (MOPD-Router, ExpertAlign) beats prompt-level hard-routing on unlabeled mixtures.

Rollout compute is the bottleneck: one student rollout + one teacher forward pass per student token. Batched systems (veRL, TRL) overlap teacher and student passes.

## Why it matters

- **The standard recipe for open-weights small models.** Qwen, Llama, DeepSeek all use OPD-style distillation from a larger sibling as the base for their small models.
- **Fixes SeqKD's exposure bias** without giving up the teacher's distributional information.
- **Composes with RL.** OPD is a supervised loss on the student's own rollout, so it can be added to a GRPO/RLVR objective as an auxiliary term — this is how many "student that reasons like the teacher" recipes work.

## Gotchas & tricks

- **Distribution drift.** As the student improves, its rollouts change; the teacher's supervision on those new contexts may itself be off-distribution for the teacher. Some recipes re-mix in offline teacher traces.
- **Reverse vs forward KL.** Forward KL ($p_T \| p_S$) is mode-covering: student hedges across teacher's high-prob tokens. Reverse KL is mode-seeking: student picks *one* teacher mode and commits. Modern reasoning distillation often uses reverse KL to avoid diluted outputs.
- **Latent alignment collapses.** Adding intermediate-layer KL can improve for ~10 steps then degrade catastrophically — the "better alignment, worse behavior" trap. Apply latent signal only at the last hidden layer or short-crossfade into token-only OPD.
- **Teacher size matters more than you'd expect.** OPD from a 4B teacher into 1.7B student is often close to OPD from an 8B teacher. Diminishing returns beyond ~2–4× student size.

## Sources

- Paper: *Distilling the Knowledge in a Neural Network* — Hinton et al., 2015 — original KD.
- Paper: *On-Policy Distillation of Language Models* — Agarwal et al., DeepMind, 2024 — https://arxiv.org/abs/2306.13649
- Paper: *LastOPD: Taming Collapse in Latent On-Policy Distillation* — Yang et al., 2026 — https://arxiv.org/abs/2609.28845 (last-layer-only latent supervision).
- Paper: *MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation* — Xu et al., Shanghai AI Lab, 2026 — https://arxiv.org/abs/2609.30837 (token-level teacher routing).
