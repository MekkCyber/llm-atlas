# AIR Schedule

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Affine Inverse-square-Root decay. A pretraining LR schedule derived from a principled analysis of the *effective learning rate* (ELR = nominal LR / weight norm) in weight-normalized training. The schedule follows `1/√t` and is affine-adjusted to meet prescribed peak and terminal LRs over a finite horizon. On 128B-token MoE pretraining, AIR beats cosine, WSD, and other candidates on terminal loss, downstream score, and hyperparameter robustness.

**Prereqs:** [_lr-schedules](_lr-schedules.md), [_training-stability](_training-stability.md)
**Related:** [wsd-schedule](wsd-schedule.md), [mid-training](mid-training.md)

---

## What it is

An LR schedule for pretraining that treats *effective* learning rate — the nominal LR divided by the current weight norm — as the first-class scheduling target. The paper's claim: previously reported gains from switching to weight-normalized (WN) optimizers over weight-decayed (WD) ones are driven almost entirely by the ELR trajectory WN induces; matching ELR between WD and WN runs collapses their training dynamics.

AIR is the schedule derived from that principle.

## How it works

Starting from a one-pole gradient-flow analysis of WN training, the paper derives that a principled ELR curve follows `1/√t` (equivalently, nominal LR times a weight-norm correction). To make this usable at a fixed training budget, AIR applies an affine adjustment so the schedule hits a user-chosen peak `η_peak` after warmup and a user-chosen terminal `η_final` at step `T`:

```
η_AIR(t) = a / sqrt(t + b)       for warmup < t ≤ T
```

with `(a, b)` solved from the two endpoint constraints. The schedule is monotonically decreasing, smooth, has no hyperparameter to tune beyond the endpoints, and provably recovers the WN gain when applied to WD.

## Why it matters

- Nominal-LR schedules (cosine, WSD) have no account of weight norm — they are the same schedule whether the parameter norms are blowing up or shrinking. AIR fixes that.
- On a 128B-token MoE pretrain, AIR reduces terminal training loss by ~0.01 vs cosine at matched cumulative LR, improves downstream scores, and is more robust to peak-LR choice. These are the kinds of gains frontier runs chase.
- Replaying AIR's ELR trajectory on a weight-decayed run reproduces WN-like dynamics — so the paper identifies the mechanism, not just a better schedule.

## Gotchas & tricks

- Benefit is largest when weight norms are drifting meaningfully (MoE training, large batches, long horizons). On short runs with tight weight decay the WD → WN difference is small, so AIR's gain over cosine is smaller.
- Terminal LR must be chosen with the data curriculum in mind. The last few percent of training commits weights disproportionately — AIR does not change that fact, it just controls *which* nominal/effective LR you commit at.
- Interacts with weight-decay strength: if you drop weight decay to zero, the `1/√t` nominal shape and the ELR shape coincide, so AIR looks like a plain `1/√t`.
- Reported for Adam-family optimizers on MoE. Transfer to Lion/Muon is untested.

## Sources

- Paper: *Your LR Schedule May Not Be Principled: Effective Learning Rate Scheduling in Weight-Normalized Training* — Yu, Ma, Bin, He, ByteDance Seed, 2026 — [arXiv:2610.40362](https://arxiv.org/abs/2610.40362).
- Related: *MiniCPM* (OpenBMB, 2024) — WSD; AIR's main empirical comparator.
- Related: *Decoupled Weight Decay Regularization* — Loshchilov & Hutter, 2017 — AdamW weight-decay background.
