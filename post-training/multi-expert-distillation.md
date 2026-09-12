# Multi-Expert On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-training recipe for consolidating multiple *specialist* checkpoints — each trained on its own narrow slice (visual aesthetics, glyph rendering, editing, …) — back into one **generalist** model via on-policy distillation. Each specialist rolls out on its own domain; the generalist student samples in the same domain and matches the specialist's token distribution on-policy. Different from standard distillation because the teacher is *domain-conditional* and the student's rollouts drive the training data. Used at scale in SenseNova-U1.5 across four visual specialists.

**Prereqs:** [../post-training/_post-training](_post-training.md)
**Related:** [rejection-sampling](rejection-sampling.md), [dpo](dpo.md), [grpo](grpo.md)

---

## What it is

The "many specialists → one generalist" pattern turns up when a single model must ship multiple capabilities that resist joint training. Each capability gets its own fine-tune (aesthetic filters, text-in-image rendering, image editing, infographic generation for SenseNova-U1.5), producing $M$ specialist checkpoints. Multi-expert on-policy distillation fuses them into one policy that inherits every specialist's behavior without hosting $M$ models.

## How it works

Given generalist student $\pi_\theta$ and specialists $\{\pi^*_1, \ldots, \pi^*_M\}$:

1. **Domain-tagged prompts.** Partition the training prompts into $M$ domains, each with a tag identifying the responsible specialist.
2. **On-policy rollouts.** For prompt $p$ in domain $m$, sample completion $y \sim \pi_\theta(\cdot \mid p)$.
3. **Specialist scoring.** Score $y$ under $\pi^*_m$ — the *specialist* for that domain.
4. **Distillation loss.** Match token distributions on-policy:
   $$L = \mathbb{E}_{p, y \sim \pi_\theta}[\mathrm{KL}(\pi_\theta(\cdot \mid p, y_{<t}) \,\|\, \pi^*_m(\cdot \mid p, y_{<t}))]$$
   The student learns from its own rollouts, evaluated against the specialist relevant to the domain.
5. **Anti-forgetting replay.** Every batch samples from *all* domains so no specialist's capability collapses; per-domain quotas are tuned to keep each specialist's contribution alive.

Optional: pair with reward-model scoring on the same rollouts to bias toward high-quality specialist behavior only.

## Why it matters

- **Serving cost.** One generalist beats $M$ specialists in serving cost and latency; distillation preserves the coverage.
- **Cross-specialist transfer.** Because the student sees all domains in one training run, capabilities that were siloed in specialists start reinforcing each other (understanding transfers into generation, editing generalizes to novel operations).
- **On-policy is the key.** Off-policy distillation on specialist-generated data suffers from covariate shift (student never learns how to *recover* from mistakes it makes but the specialist would not). Rolling out with the student closes that gap.
- **Composable with unified-model architectures.** Fits naturally with MoT + native-unified-model designs where multiple modalities/specialties already live in one backbone; distillation just consolidates post-training.

## Gotchas & tricks

- **Per-domain quota tuning.** The dominant domain in the training mix wins by default; explicit quotas prevent capability collapse.
- **Specialist drift.** Specialists trained separately may disagree on shared prompts; freeze the specialists during distillation, and choose the specialist per-prompt (not per-token).
- **Reward-model blending.** Adding a reward-model term stabilizes noisy specialists; too much and you distill the reward model instead of the specialist.
- **Watch mode collapse into the dominant specialist.** If one specialist has a much sharper distribution than the others, KL will over-index on it; anneal per-domain temperature.
- **Not the same as MoE routing.** MoE routes at *inference*; here you distill many models into one policy that generates without routing.

## Sources

- Paper: *SenseNova-U1.5: Towards Native Unified Visual Intelligence* — SenseTime, 2026 — [arXiv:2609.11929](https://arxiv.org/abs/2609.11929) — introduces multi-expert on-policy distillation across four visual specialists.
- Related: standard model distillation and on-policy distillation literature.
