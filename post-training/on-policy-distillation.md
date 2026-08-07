# On-Policy Distillation (OPD)

*Depth — dense, token-level teacher supervision applied to student-sampled trajectories, sitting between SFT and RL.*

**TL;DR:** OPD transfers a teacher's capability to a student by running rollouts from the **student's** policy and, at every token, matching the student's distribution to the teacher's on the same context. It's SFT with rollouts sampled from the student instead of a fixed dataset — "on-policy" in the distillation sense — which sidesteps SFT's train/test distribution shift and gives dense per-token signal that RL lacks. Widely used as a rescue when RL loses gradient (e.g., zero-variance GRPO groups) and as a standalone recipe when teacher access is cheap.

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md)
**Related:** [sa-opd.md](sa-opd.md) · [rstg.md](rstg.md) · [rejection-sampling.md](rejection-sampling.md) · [_post-training.md](_post-training.md)

---

## What it is

Two neighboring paradigms:

- **Offline distillation (SFT-style)**: sample completions from the *teacher*, train the student to imitate. Simple, but the student never sees its own errors — trained on states it wouldn't otherwise visit.
- **On-policy distillation**: sample completions from the *student*, then ask the teacher what it would have said at each of those states. Train to match the teacher's next-token distribution.

The second is what "OPD" refers to. The signal is dense (per-token, not per-sequence), on-distribution (student's own errors are the ones being corrected), and does not require a reward function.

## How it works

For each RL-style step:

1. **Rollout** a batch of trajectories from the current student policy `π_student`.
2. **Score** each student-visited context with the teacher: obtain `π_teacher(· | context)` — either logits (if teacher is white-box) or top-k probabilities (black-box).
3. **Compute a divergence loss** at every token: typically forward KL `KL(π_teacher || π_student)` — teacher-centered, so tokens the teacher is confident about carry more weight than tokens the teacher is unsure of.
4. **Update** the student. No advantage estimation, no critic, no reward model.

Compared to GRPO, OPD trades exploration (there's no reward-driven search) for signal density (every token gets gradient). Compared to SFT, OPD trades data curation for teacher-inference cost.

## Why it matters

- **Rescues RL when signal dies.** GRPO groups where every response gets the same reward provide no gradient; OPD gives dense signal on those exact prompts. This is the primary "combine OPD with GRPO" motivation (e.g. RSTG).
- **On-distribution.** SFT teaches "what the teacher does on ideal states"; OPD teaches "what the teacher would do on the states the student actually reaches" — the latter is what closes the deployment gap.
- **Reward-free.** Works whenever teacher access exists, no verifier or reward model needed.

## Gotchas & tricks

- **Spurious teacher signals.** Some teacher outputs are driven by language priors or formatting conventions, not by the input — these produce large gradients that don't improve task behavior. [sa-opd.md](sa-opd.md) is the standard fix: filter tokens with low input-groundedness and high divergence.
- **Kills exploration if stacked on GRPO naively.** OPD sharpens the student toward the teacher; if applied to *all* GRPO samples (not just failing ones), the student stops exploring — the [rstg.md](rstg.md) restriction to negative-zero-variance prompts is the standard fix.
- **Asymmetric advantage.** Because forward KL is teacher-weighted, OPD tends to suppress most tokens and boost a few — the resulting policy can become peaked and lose diversity.
- **Teacher-inference cost dominates.** Every student rollout requires teacher forward passes on the *same* contexts. For a 100B teacher × millions of tokens, this is the main compute expense.
- **Black-box teachers work.** Top-k probabilities from an API are enough; you don't need full logits. But sparse targets bias the loss slightly.

## Sources

- Paper (SA-OPD): *When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation* — Jiang, Ye, Tao, Zhuang, Zhang, Chen, Li, 2026 — [arXiv 2608.03632](https://arxiv.org/abs/2608.03632). Frames OPD's spurious-signal failure mode.
- Paper (RSTG): *Distill Where You Fail: Recovering Learning Signals of Negative RL-Groups from Adaptive Teacher Guidance* — Han et al., 2026 — [arXiv 2608.00782](https://arxiv.org/abs/2608.00782). GRPO + OPD stacking with selective application.
- Related: *DistiLLM* / *MiniLLM* / *Sequence-Level KD* — earlier lines that motivated the on-policy variant.
