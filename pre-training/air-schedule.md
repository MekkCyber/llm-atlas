# AIR: Affine Inverse-square-Root LR Decay
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A learning-rate schedule designed around the *effective* learning rate (ELR) rather than the nominal LR. The core shape is a theoretically-derived $1/\sqrt{t}$ decay, adjusted by an affine map so that the ELR trajectory hits a prescribed peak and terminal value over a fixed training horizon. The paper's framing claim: once you match ELR, weight-normalized (WN) and weight-decayed (WD) training have essentially the same loss dynamics — so schedule *design* should target ELR, not nominal LR. On 128B-token MoE pretraining, AIR beats cosine on training loss, downstream, and hyperparameter robustness by ~0.01 terminal loss at matched cumulative LR.

**Prereqs:** [_lr-schedules](_lr-schedules.md)
**Related:** [_training-stability](_training-stability.md), [wsd-schedule](wsd-schedule.md), [mid-training](mid-training.md)

---

## What it is

The *effective learning rate* at step $t$ under weight normalization is approximately

$$
\eta_{\text{eff}}(t) \approx \frac{\eta(t)}{\|w(t)\|^2}
$$

where $\eta(t)$ is the nominal LR and $w(t)$ is the (normalized) weight. Under WD without WN, $\|w(t)\|$ follows a different trajectory, so the same nominal schedule produces different ELRs. AIR *inverts the mapping*: pick the ELR trajectory you want, then derive the nominal schedule that produces it.

The derived shape is $1/\sqrt{t}$ (from a theoretical analysis of the ELR dynamics under WN). An affine transform in time and value fits two boundary conditions:

$$
\eta_{\text{AIR}}(t) = \frac{A}{\sqrt{t + B}} + C, \quad A, B, C \text{ chosen so } \eta_{\text{eff}}(0) = \eta_{\max},\; \eta_{\text{eff}}(T) = \eta_{\text{end}}
$$

## How it works

- **Target the ELR trajectory, not nominal LR.** Pick $(\eta_{\max}, \eta_{\text{end}}, T)$ for the ELR; solve for $(A, B, C)$ in the nominal schedule.
- **Shape is $1/\sqrt{t}$**, affine-adjusted. Not a cosine, not a WSD step.
- **Late ELR dominates.** The paper empirically shows the final fraction of the ELR trajectory controls most of the schedule's downstream effect. A shared late schedule attenuates earlier LR choices.
- **Equivalence under WD.** Applying the same ELR trajectory to a WD run reproduces the WN dynamics. This is the strongest form of the authors' claim: WN's win over WD is really an ELR-trajectory win, not a normalization win.

## Why it matters

Most published LR schedules are shape choices (cosine, WSD, linear) without a principled target. AIR gives a *scheduling object* (ELR) and a *shape* ($1/\sqrt{t}$) that falls out of the dynamics. In the paper's experiments:

- **128B-token MoE pretraining:** AIR reduces terminal training loss by ~0.01 relative to cosine at matched cumulative LR.
- **Hyperparameter robustness:** AIR is less sensitive to peak-LR choice than cosine.
- **WN↔WD equivalence under matched ELR:** collapses a long-running comparison.

Likely effect on the field: future schedule comparisons report ELR curves alongside nominal LR, and WN-vs-WD benchmarks become ELR-matched rather than LR-matched.

## Gotchas & tricks

- **ELR depends on $\|w\|$**, which in turn depends on the training dynamics. The paper's $1/\sqrt{t}$ is a derived target that holds *approximately*; large deviations in weight-norm trajectory may require re-fitting the affine terms mid-run.
- **The late-ELR result is actionable on its own.** Even if you keep cosine, matching the terminal fraction of the ELR trajectory across runs removes most of what previously looked like schedule-shape differences.
- **At matched cumulative LR**, not just matched peak LR — the paper's cosine comparison is apples-to-apples under the integral.
- **MoE results.** Reported on MoE pretraining; dense scaling behavior is plausible but not demonstrated at the same scale.

## Sources

- Paper: *Your LR Schedule May Not Be Principled: Effective Learning Rate Scheduling in Weight-Normalized Training* — Yu, Ma, Bin, He, ByteDance Seed, 2026 — arXiv:2610.40362.
