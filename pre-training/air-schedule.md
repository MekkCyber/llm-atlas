# AIR: Affine Inverse-square-Root LR Schedule
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Under weight normalization (RMSNorm, QK-norm, etc.) the *effective* LR felt by the parameters depends on current weight norm, not just the set LR. AIR (**A**ffine **I**nverse-square-**R**oot) hand-shapes this effective-LR trajectory as a classical $1/\sqrt{t}$ decay affinely stretched to hit chosen peak and terminal values over a fixed token budget. On 128B-token MoE pretraining AIR beats cosine on loss, downstream, and hyperparameter robustness.

**Prereqs:** [_lr-schedules.md](_lr-schedules.md)
**Related:** [wsd-schedule.md](wsd-schedule.md), [_training-stability.md](_training-stability.md), [../architectures/qk-norm.md](../architectures/qk-norm.md)

---

## What it is

In weight-normalized training, if a weight tensor $w$ has norm $\|w\|$, a gradient step of size $\eta$ moves the normalized direction $w/\|w\|$ by roughly $\eta/\|w\|$. The *effective learning rate* (ELR) is thus $\eta / \|w\|$ — and $\|w\|$ drifts with training. Cosine and WSD schedules shape $\eta(t)$, but the ELR trajectory under them is not itself cosine or WSD-shaped. AIR shapes the ELR *directly*.

## How it works

Start from the classical inverse-square-root schedule from stochastic approximation theory:

$$
\mathrm{ELR}(t) = \frac{c}{\sqrt{t}}
$$

This is infinite-horizon and has no notion of a budget. AIR applies a two-parameter affine fit so that over a fixed horizon $T$ the ELR hits a chosen peak $\mathrm{ELR}_{\mathrm{peak}}$ at step $t_0$ (post-warmup) and a chosen terminal $\mathrm{ELR}_{\mathrm{end}}$ at step $T$:

$$
\mathrm{ELR}(t) = a + \frac{b}{\sqrt{t}}
$$

with $a, b$ solved from the two endpoint constraints. The *set* LR $\eta(t)$ is then computed from $\mathrm{ELR}(t) \cdot \|w(t)\|$ with $\|w(t)\|$ either tracked online or approximated from a known normalization-induced trajectory.

In practice: warmup linearly, then follow AIR's affine-inverse-sqrt curve on the ELR. The paper evaluates this on 128B-token MoE pretraining and reports terminal training loss reduced by ~0.01 nats vs. cosine at matched cumulative LR, with downstream benchmark lift and lower sensitivity to the peak-LR choice.

## Why it matters

Most LLM recipes paper over the schedule-×-normalization interaction by brute-force tuning cosine's peak LR. AIR separates the concerns: pick the normalization for its optimization geometry, then shape the ELR directly. Especially relevant for MoE and long-context runs where weight-norm drift is pronounced. The theoretical grounding (classical SA inverse-sqrt plus a budget-aware affine fit) makes it harder to over-tune.

## Gotchas & tricks

- **Need to track or model $\|w\|$.** The simplest implementation records the dominant weight norm and divides at optimizer step time; the paper also offers an analytic approximation.
- **Warmup is still linear.** AIR replaces the main body, not the warmup.
- **Peak LR interpretation changes.** The "peak" you set is the peak *effective* LR, not the raw $\eta$ you'd set under cosine.
- **MoE specifically.** The paper argues gains are larger at MoE scale because routing-induced weight dynamics magnify ELR drift.
- **Match the data curriculum.** Like WSD, AIR's terminal ELR lands in the critical "commit" window; align the terminal data mix with it.

## Sources

- Paper: *Your LR Schedule May Not Be Principled: Effective Learning Rate Scheduling in Weight-Normalized Training* — Yu, Ma, Bin, He — ByteDance Seed, 2026 — [arXiv:2610.40362](https://arxiv.org/abs/2610.40362).
- Paper: *Attention Is All You Need* — Vaswani et al., 2017 — inverse-sqrt as original schedule.
- Reference: Robbins & Monro's stochastic approximation conditions give the $1/\sqrt{t}$ ancestry.
