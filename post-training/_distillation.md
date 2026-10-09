# Distillation for LLM Post-Training

*Taxonomy — the family of techniques that transfers knowledge from a stronger model (or an older checkpoint) into a student policy during post-training.*

**TL;DR:** Classical distillation picks a frozen teacher and trains the student to match its output distribution on a fixed dataset. Modern LLM post-training distillation is **on-policy**: the student samples its own rollouts, and the teacher supervises *those* samples token-by-token or with contrastive anti-signal. The two newest wrinkles — **negative-policy** rollouts and **self-retrospection** signals — recover gradient in regimes where plain on-policy distillation collapses.

**Related taxonomies:** [_rl](_rl.md) · [_post-training](_post-training.md) · [_rewards](_rewards.md)
**Depth files covered here:** [on-policy-distillation](on-policy-distillation.md) · [np-opd](np-opd.md) · [self-retrospection-distillation](self-retrospection-distillation.md) · [rejection-sampling](rejection-sampling.md)

---

## The problem

Pure SFT on a teacher's responses locks the student to the teacher's output distribution — only the prompts the teacher happened to answer, only the tokens the teacher happened to pick. The student never sees its own failure modes. RLVR fixes the "its own failure modes" part but silently fails whenever the reward is coarse (all rollouts in a group get the same scalar). Distillation during post-training is the bridge: dense teacher signal, applied to **the student's own rollouts**.

## The shared pattern

Every modern distillation variant has the same three slots:

1. **Who samples?** Student (on-policy), teacher (off-policy), or a mix (NP-OPD adds a negative policy).
2. **What is the signal?** Token-level KL, logit matching, or a hindsight-vs-foresight contrast.
3. **What is the loss doing?** Pulling the student toward (positive) or pushing it away from (negative) the referenced distribution.

Framed as RL, the teacher is an **implicit reward model** over the student's rollouts (see `on-policy-distillation.md`). Everything else is a choice of reward class.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| SFT distillation | Student imitates teacher outputs on fixed prompts | Student stuck in teacher's distribution | Cheap bootstrap; cold start |
| [rejection-sampling](rejection-sampling.md) | Sample K; filter with verifier; SFT survivors | Needs a good filter | Verifiable tasks (math, code) |
| [on-policy-distillation](on-policy-distillation.md) | Student samples; teacher scores each token | Collapses to long/repetitive outputs | Capability transfer without a verifier |
| [np-opd](np-opd.md) | Add a weaker "negative" policy; push away | One more inference stream | Breaks OPD collapse |
| [self-retrospection-distillation](self-retrospection-distillation.md) | Hindsight model → foresight model, same weights | Needs a trajectory to look back on | Agentic tasks, scarce reward contrast |

## How to choose

- **Have a reliable verifier?** Rejection sampling first — it's the cheapest and composes with everything downstream.
- **Have a stronger teacher but no verifier?** On-policy distillation. Watch for collapse: use SFT-init and mask degenerate rollouts.
- **OPD collapsing?** Add a negative policy (NP-OPD). It is nearly free.
- **No teacher, no verifier, agentic rollouts?** Self-retrospection distillation. The hindsight copy of your own policy provides the signal.

## Adjacent but distinct

- [_rl](_rl.md) — policy gradient methods. Distillation ⊂ RL under the "teacher is a reward model" framing.
- [cot-reward-model](cot-reward-model.md) — a different *source* of reward signal, not a distillation variant.
- Weight-level distillation (Minitron-style pruning + KL on a frozen teacher) — concerns pretraining-scale compression, not post-training.

## Sources

- *Gains and Collapse in On-Policy Distillation* — Cui et al., 2026 — [arXiv:2610.03185](https://arxiv.org/abs/2610.03185)
- *On-Policy Distillation with Negative-Policy Rollouts* — Han et al., 2026 — [arXiv:2610.07874](https://arxiv.org/abs/2610.07874)
- *Self-Retrospection Distillation* — Zhang et al., 2026 — [arXiv:2610.08077](https://arxiv.org/abs/2610.08077)
