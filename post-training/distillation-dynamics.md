# Distillation Dynamics (On- vs Off-Policy, KL Direction, LR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A controlled strong-to-weak distillation study that disentangles three axes usually bundled together — rollout policy (on-policy vs off-policy), token-level KL direction (forward vs reverse), and learning rate. The surprising takeaway: forward KL is remarkably robust to rollout policy, reverse KL is not, and *learning rate* dominates the "catastrophic forgetting" story usually blamed on off-policy data.

**Prereqs:** [_rl.md](./_rl.md), [rlvr.md](./rlvr.md), [rejection-sampling.md](./rejection-sampling.md)
**Related:** [grpo.md](./grpo.md), [dpo.md](./dpo.md)

---

## What it is

A systematic ablation (Piskorz, Berthon, van der Schaar; Cambridge, 2026) of what actually matters in strong-to-weak LLM distillation. The field had been running the comparison "SFT vs RL" with many factors varying at once; this paper fixes the task, model, and data, and sweeps three axes independently: (a) rollout policy on a continuous student↔teacher spectrum, (b) forward vs reverse token-level KL, (c) learning rate. Measured on Llama-3 and Qwen-2.5 across scientific, medical, and arithmetic reasoning tasks.

## How it works

- **Rollout policy spectrum.** Instead of binary on/off-policy, the student samples rollouts from a mixture $\pi_\alpha = \alpha \pi_\text{student} + (1-\alpha) \pi_\text{teacher}$. Sweep $\alpha$ from 0 to 1.
- **KL direction.** The distillation loss is a token-level KL between student and teacher logits — either forward $\text{KL}(p_\text{teacher} \| p_\text{student})$ (mode-covering) or reverse $\text{KL}(p_\text{student} \| p_\text{teacher})$ (mode-seeking).
- **Learning rate.** Swept across the usual range for each setting.
- **Gradient analysis.** For forward KL, the gradient magnitude is approximately insensitive to which policy generated the rollout; for reverse KL, the gradient vanishes on tokens where the student assigns zero mass, which is exactly where teacher-generated rollouts land, so reverse KL favours student rollouts.

## Why it matters

- Reframes "SFT bad / on-policy good." Cheap off-policy teacher rollouts with forward KL remain a strong default; the places where on-policy really pays off are narrower than the folklore suggests.
- Identifies **learning rate** — not rollout policy — as the primary driver of both catastrophic forgetting and update sparsity. Lets practitioners tune the right knob.
- Gives a mechanism (gradient-vanishing on OOD tokens) for why reverse KL is so sensitive to rollout policy. Transfers directly to KD-style auxiliary losses inside RLHF/RLVR pipelines.

## Gotchas & tricks

- On-policy data *does* help generalisation on harder Countdown variants, but the advantage **does not survive a subsequent RLVR stage** — if you RL afterward, your choice of distillation rollout policy mostly washes out.
- The claims hold with and without gradient clipping, with sampled KL estimators, and on longer-reasoning tasks — the authors explicitly stress-test the headline.
- "Reverse KL + off-policy" is the specific combination that fails hardest. If you must use reverse KL (e.g. for mode-seeking behavior), budget for student rollouts.
- Forgetting looks catastrophic at the usual SFT LRs ($\sim 10^{-5}$) and benign at the lower RL LRs ($\sim 10^{-6}$) — some of what papers attribute to SFT-vs-RL is actually LR-vs-LR.

## Sources

- Paper: *On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics* — Piskorz, Berthon, van der Schaar, 2026 — [arXiv:2609.35259](https://arxiv.org/abs/2609.35259).
- Context: *DeepSeekMath / DeepSeek-R1* — DeepSeek, 2024–2025 — the RLVR stage that erases the distillation rollout-policy effect.
- Context: *MiniLLM* — Gu et al., 2023 — reverse-KL distillation that this paper's reverse-KL branch builds on.
