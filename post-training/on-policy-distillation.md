# On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** **On-policy distillation** trains a student LLM using its own generations as inputs, supervised token-by-token by a teacher's next-token distribution. Unlike off-policy distillation (SFT on teacher-generated text), the student sees its own distribution and gets teacher gradients exactly where it matters — on tokens it would actually generate. Cheaper than RL for many post-training targets, and higher-fidelity than off-policy SFT.

**Prereqs:** [_post-training.md](_post-training.md), [rejection-sampling.md](rejection-sampling.md)
**Related:** [delta-distillation.md](delta-distillation.md), [dpo.md](dpo.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

The distillation family:

| Variant | Rollout source | Supervision signal | Cost |
| --- | --- | --- | --- |
| **Off-policy (SFT-on-teacher)** | Teacher generates responses | Cross-entropy on teacher tokens | Cheap; may mismatch student distribution |
| **Off-policy soft distillation** | Teacher generates | KL(student ‖ teacher) at each token | Same rollout cost + teacher logits |
| **On-policy hard distillation** | **Student generates**, teacher scores | Cross-entropy on teacher's argmax | Student rollout + teacher forward |
| **On-policy soft distillation** | **Student generates**, teacher scores | KL(student ‖ teacher) at each student token | Same as above; richer signal |

On-policy variants match RL in one crucial way: the training distribution equals the deployment distribution (both are student generations). Off-policy variants train on teacher generations, which the student may never produce at deploy — a distribution shift that costs quality.

## How it works

1. **Student generates.** For each prompt $q$, sample a rollout $o = (o_1, \ldots, o_T)$ from the student policy $\pi_\theta$.
2. **Teacher scores every token.** Run the teacher $\pi_T$ forward on the same $(q, o_{<t})$ prefixes to get $\pi_T(\,\cdot\, \mid q, o_{<t})$ for every position $t$.
3. **Loss.** Either cross-entropy on teacher's argmax token (hard) or KL between full distributions (soft):
   $$L = \sum_t \mathrm{KL}\big(\pi_T(\,\cdot\, \mid q, o_{<t}) \,\|\, \pi_\theta(\,\cdot\, \mid q, o_{<t})\big)$$
4. **Backprop into student only.** Teacher is frozen.

The cost per step is one student rollout + one teacher forward pass, comparable to on-policy RL with a reward model. But there's no advantage estimation, no clipping, no reward hacking — just token-level supervision on the student's own distribution.

## Why it matters

- **Bridges RL and SFT.** On-policy soft distillation gives token-level supervision (like SFT) on the student's own outputs (like RL), collecting the strengths of both.
- **No reward model needed.** The teacher's distribution is the signal — sidesteps reward-model construction, calibration, and hacking issues.
- **Faster convergence than RL** in domains where the teacher is strong and available. RL still wins when the teacher is weak or when the reward is not encoded anywhere in the teacher's distribution.
- **Compositional with delta signals.** OPD² ([delta-distillation.md](delta-distillation.md)) shows that supervising the *delta* between reasoning-tuned and base teacher produces cleaner reasoning transfer than raw distribution matching.

## Gotchas & tricks

- **Teacher must be able to run per-step forwards on student rollouts.** Two large models loaded at once; that's the compute cost.
- **KL vs. cross-entropy is a signal-richness tradeoff.** KL uses the full teacher distribution (richer, slightly more compute); hard-label CE is simpler and often within a small delta on quality.
- **Rollout length affects what you distill.** Long rollouts with tiny per-token gradients can waste compute; consider truncating to informative prefixes.
- **Watch for teacher collapse regions.** Where the teacher is nearly deterministic, KL saturates; the student learns nothing new. Mix in RLVR or SFT on the same prompts to keep signal flowing.
- **Combines with SFT warmup.** Cold-start student → SFT on teacher data → then on-policy distillation is a standard recipe.

## Sources

- Paper: *On-Policy Delta Distillation* — NAVER AI, 2026 — [arXiv:2607.15161](https://arxiv.org/abs/2607.15161).
- Foundational: *Distilling the Knowledge in a Neural Network* — Hinton, Vinyals, Dean, 2015 — the classical off-policy formulation.
- Reference: *DistilBERT / DistilGPT* — Sanh et al., 2019 — canonical off-policy distillation for LLMs.
