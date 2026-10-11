# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation trains a student LLM by minimizing a KL divergence between the student's own on-policy samples and a teacher's distribution. The *direction* of the KL matters: **reverse KL** OPD transfers compositional **skills** (reasoning structures the student can apply to new inputs) but barely moves **factual knowledge**, while **forward KL** restores knowledge transfer at the cost of some skill fidelity. Shown in a controlled synthetic study and confirmed on factual QA and competition-math benchmarks.

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md), [reasoning/online-policy-mirror-descent.md](reasoning/online-policy-mirror-descent.md)
**Related:** [fine-tuning/README.md](fine-tuning/README.md) · [_post-training.md](_post-training.md)

---

## What it is

Distillation has two canonical flavors. *Off-policy* distillation trains the student on a fixed teacher dataset (teacher-generated text + teacher logits). *On-policy* distillation samples text from the student itself, then matches teacher behavior on those samples — this is what RL-style training looks like with a reward replaced by teacher–student divergence. The paper disentangles what OPD transfers as a function of KL direction.

## How it works

- **Student samples.** Draw $o \sim \pi_\text{student}(\cdot \mid q)$ for each prompt $q$.
- **Divergence objective.** Compute a per-token or per-sequence divergence between student and teacher.
  - **Reverse KL:** $\mathrm{KL}(\pi_\text{student} \,\|\, \pi_\text{teacher})$ — mode-seeking.
  - **Forward KL:** $\mathrm{KL}(\pi_\text{teacher} \,\|\, \pi_\text{student})$ — mode-covering.
- **Gradient step.** Standard policy gradient on the divergence; keep the student close to a reference via KL, same as RL.

## Why it matters

- Clarifies the common "distillation gives SFT-quality results but on policy" claim by showing that *what* is transferred depends on KL direction.
- Reverse KL is the right choice when the teacher has skills the student lacks on structures the student will encounter; forward KL is the right choice when the teacher knows facts the student should learn.
- Mix the two (two-term loss) to shape both axes; the paper shows the pure variants as endpoints.

## Gotchas & tricks

- Reverse KL under-explores: if the student's initial support misses the teacher's modes, it will never find them without additional entropy or exploration.
- Forward KL is heavier to compute (need the full teacher distribution on student's samples) and more sensitive to teacher log-prob tails.
- "Factual QA" in the paper is a controlled benchmark; transfer behavior on open-domain QA with retrieval is not characterized.
- The skills-vs-knowledge split is not binary: compositional reasoning can smuggle in facts, and vice versa. Treat the paper's finding as a strong tendency, not a dichotomy.

## Sources

- Paper: *On-Policy Distillation Teaches New Skills but Not New Knowledge* — Tang, Yang, 2026 — [arXiv:2610.09639](https://arxiv.org/abs/2610.09639).
