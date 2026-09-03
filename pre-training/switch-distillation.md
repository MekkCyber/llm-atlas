# Switch Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **mid-training-specific** knowledge-distillation objective that gates per-token between forward-KL from a stronger teacher (when the teacher is confident) and plain cross-entropy with the ground-truth token (when it isn't). Fixes a stage-dependent regression: standard forward-KL KD *helps* reasoning + factual recall during pre-training, but *slows* factual recall during mid-training. Switch Distillation reaches 1.61–1.71× the reasoning of NTP while preserving 96.7–96.8% of factual recall.

**Prereqs:** [mid-training.md](mid-training.md)
**Related:** [_lr-schedules.md](_lr-schedules.md), [wsd-schedule.md](wsd-schedule.md), [model-souping.md](model-souping.md)

---

## What it is

Logit-based knowledge distillation trains a smaller student on the *soft* output distribution of a stronger, usually post-trained, teacher. The standard formulation is per-token forward-KL:

$$
\mathcal{L}_{\text{FKL}} = \sum_t \mathrm{KL}\!\left( p_{\text{teacher}}(\cdot \mid x_{<t}) \,\|\, p_{\text{student}}(\cdot \mid x_{<t}) \right)
$$

Empirically, forward-KL helps both reasoning and factual recall during pre-training relative to standard next-token prediction (NTP). During **[mid-training](mid-training.md)** — the annealed, high-quality-mix phase — the same objective flips: reasoning still improves, but factual recall regresses vs. NTP.

Switch Distillation is a per-token gate that fixes this:

$$
\mathcal{L}_{\text{Switch}} = \sum_t
\begin{cases}
\mathrm{KL}\!\left( p_{\text{teacher}} \,\|\, p_{\text{student}} \right) & \text{if } H(p_{\text{teacher}}) < \tau \\
-\log p_{\text{student}}(x_t \mid x_{<t}) & \text{otherwise}
\end{cases}
$$

Teacher predictive entropy is the routing signal.

## How it works

Two asymmetries drive the fix:

1. **Teachers are more confident on procedural than knowledge-intensive tokens.** A post-trained teacher's forward-KL is a strong signal for reasoning steps (procedure), but on rare-fact tokens it's diffuse and effectively dilutes the ground-truth signal.
2. **Students acquire low-entropy factual knowledge earlier in training.** By mid-training, the student's own probability on many factual tokens is already sharper than the teacher's dispersion, so distillation on those tokens overwrites correct student mass with vague teacher mass.

Switch Distillation exploits (1) by *keeping* forward-KL where it helps (low-entropy teacher, procedural tokens) and swapping to cross-entropy where it hurts (high-entropy teacher, knowledge tokens). The gate is a single threshold on teacher entropy — no new networks.

## Why it matters

- **Fixes a hidden regression in the default mid-training KD recipe.** Any lab distilling into small models during mid-training was leaving factual recall on the table without noticing, because reasoning benchmarks improve either way.
- **One-hyperparameter drop-in.** Entropy threshold $\tau$ is the only knob added; no new networks, no gradient accountings, no scheduling.
- **Gains persist through post-training.** After downstream SFT/RL, the switch-distilled student keeps 1.25–1.32× reasoning and 1.13–1.20× knowledge/commonsense over an NTP baseline, and closes the factual-recall gap.
- **Cross-teacher robust.** Reported gains hold across a range of teacher sizes.

## Gotchas & tricks

- **This is mid-training-specific.** During pre-training, plain forward-KL still helps both axes — don't switch objectives there.
- **Entropy threshold $\tau$ needs calibration.** The paper reports a working default but the value depends on teacher size and vocabulary. Sweep on a held-out slice.
- **Not the same as sample-level KD gating.** The switch is per-token, so the same sequence can carry both KL and CE tokens; gating whole sequences loses most of the gain.
- **Teacher must be post-trained.** The confidence-asymmetry story assumes a teacher that has been RLHF'd or SFT'd on procedure-heavy data. A pretraining-only teacher won't show the same entropy profile.
- **The gate is asymmetric on failure.** Missing the switch (staying in KL on high-entropy tokens) is the *original* failure mode. Missing the other direction (falling back to CE on low-entropy tokens) mostly no-ops.

## Sources

- Paper: *Knowledge Distillation During Mid-Training Favors Reasoning over Factual Recall* — He et al., 2026 — [arXiv:2609.01532](https://arxiv.org/abs/2609.01532).
