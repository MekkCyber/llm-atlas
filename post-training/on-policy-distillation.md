# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A distillation recipe where the **student** samples its own rollouts from the current policy, then a **frozen teacher** provides dense token-level supervision (usually a KL against the teacher's next-token distribution) over the student's own trajectories. Unlike offline distillation on a fixed teacher-generated dataset, the supervision matches the distribution the student actually visits — which stabilizes small-model post-training and pairs well with RL. The "self" variant (OPSD) uses a frozen copy of the *same* model, often with privileged information (an answer or worked solution) fed into the teacher's context.

**Prereqs:** [rejection-sampling.md](./rejection-sampling.md), [_post-training.md](./_post-training.md)
**Related:** [grpo.md](./grpo.md) · [rlvr.md](./rlvr.md) · [dpo.md](./dpo.md)

---

## What it is

Classical knowledge distillation collects a dataset of teacher outputs and trains a student to imitate them token-by-token. That's *off-policy*: the student is supervised on the teacher's trajectories, not its own. When the student later generates on its own it visits states the teacher never showed — and the imitation loss said nothing about those states.

On-policy distillation flips this: **the student generates, the teacher scores**. For each student rollout, the teacher (a frozen model, larger and/or post-trained) is run over the same prefix and its next-token distribution becomes the target of a KL divergence against the student's. Every token the student emits gets a dense, distribution-level supervision signal — no reward model, no sparse trajectory return.

## How it works

For each prompt $q$, sample a rollout $o$ from the current student $\pi_\theta$. Run the teacher $\pi_T$ over $(q, o_{<t})$ at every position. The loss is a per-token KL:

$$
L_{\text{OPD}} = \frac{1}{|o|}\sum_t \mathrm{KL}\!\left( \pi_T(\cdot \mid q, o_{<t}) \,\|\, \pi_\theta(\cdot \mid q, o_{<t}) \right)
$$

Optionally combine with an RL objective (GRPO, RLVR) on the same rollouts — dense OPD signal + sparse verifiable reward. Two knobs matter:

- **Rollout mode.** Direct-response rollouts vs. thinking-enabled (long-CoT) rollouts change what the KL is fit against. Recent work shows the *same* teacher/reference view can help direct-response rollouts and hurt thinking rollouts at the same checkpoint.
- **Privileged teacher context.** OPSD often feeds the teacher an answer, a worked solution, or a golden trace so its next-token distribution is sharper. What the "privileged" reference *adds* over reference-free OPD is modest and view-dependent (see AMPLE-Math).

## Why it matters

- **Dense supervision without a reward model.** Fits between SFT (cheap, off-policy) and RLHF/GRPO (dense reward-driven). Especially useful in agentic RL where trajectory rewards are sparse and a token-level teacher speeds credit assignment.
- **Self-distillation route.** With OPSD, the teacher is *a copy of the model itself* — no larger checkpoint required. The gain is largely from re-accessing the model's own capabilities across generation modes (direct-response ↔ thinking) rather than from novel knowledge.
- **Composes with RL.** RetireOPD (2026) runs GRPO and OPD jointly, then adaptively retires the teacher when the student saturates — student surpasses teacher in every configuration tested.

## Gotchas & tricks

- **EOS-token disagreement inflates length.** Base student and post-trained teacher can place stopping mass on different EOS-like sentinels (`<|endoftext|>` vs `<|im_end|>`). If declared stopping sets match but internal preferences don't, KL suppresses the student's natural stop without transferring the teacher's — outputs grow until they hit the budget. Fix: treat functionally equivalent EOS tokens as one shared stopping action before computing the loss.
- **Schedule matters.** Fixed-schedule teacher weights leave gains on the table. Adaptive retirement (drop the teacher when discrepancy plateaus and student hits a fraction of teacher success rate) is the current best practice.
- **View mismatch flips the sign.** A privileged reference that helps at direct-response training can *hurt* long thinking-enabled rollouts at the same checkpoint. Match the rollout mode to how the model will be evaluated.
- **Late-training length drift.** Even after termination alignment, a second inflation source emerges late in OPD runs. It's not fully understood — cap total OPD tokens and inspect length curves.

## Sources

- Paper: *When EOS Tokens Disagree: Understanding Length Inflation in On-Policy Distillation* — Yang et al., UNC + Microsoft, 2026 — arXiv:2609.20511.
- Paper: *RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning* — Yu et al., Zhejiang, 2026 — arXiv:2609.20784.
- Paper: *What Does Privileged Information Add to On-Policy Self-Distillation?* — Zhang et al., NUS, 2026 — arXiv:2609.20612.
