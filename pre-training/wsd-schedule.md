# WSD — Warmup, Stable, Decay
*Depth — a three-phase LR schedule that separates capability building from "committing", enabling cheap re-runs of the final phase.*

**TL;DR:** Instead of cosine-decaying the LR across the whole run, hold LR **constant at peak** for the bulk of training ("stable" phase), then decay it sharply to zero in a short final phase ("decay"). The stable checkpoint becomes a **reusable artifact**: you can fork it and run multiple decay-phase experiments with different data mixes, saving one model but comparing many final products. Popularized by MiniCPM, now common at labs that iterate on mid-training data.

**Prereqs:** [transformer-block](../architectures/transformer-block.md)
**Related:** [mid-training](mid-training.md), [model-souping](model-souping.md), [_lr-schedules](_lr-schedules.md), [olmo-2 case study](../case-studies/olmo-2.md)

---

## What it is

Three phases with trivial math:

```
LR(step) =
    peak * (step / warmup_steps)                          if step < warmup_steps
    peak                                                  if warmup_steps ≤ step < decay_start
    peak * decay_fn((step - decay_start) / decay_steps)   if decay_start ≤ step < end
    0                                                     otherwise
```

where `decay_fn` is typically linear (`1 - t`) or 1-√t or cosine. No hyperparameter for "how sharp is the cosine" — you just pick the start and end points of the decay.

Typical proportions:

| Phase | Tokens | Notes |
|---|---|---|
| Warmup | 0.5–2% | Standard linear ramp to peak LR |
| Stable | 80–95% | LR pinned at peak, bulk of training |
| Decay | 5–20% | LR → 0; aligned with [mid-training](mid-training.md) mix swap |

Compare to cosine, which has no "stable" — LR starts decaying immediately after warmup and monotonically falls over the entire run.

### PyTorch scheduler

```python
import torch
from torch.optim.lr_scheduler import LambdaLR

def wsd_lambda(step, warmup_steps, stable_end_step, decay_end_step):
    if step < warmup_steps:
        return step / warmup_steps                                # warmup
    if step < stable_end_step:
        return 1.0                                                # stable at peak
    if step < decay_end_step:
        progress = (step - stable_end_step) / (decay_end_step - stable_end_step)
        return 1.0 - progress                                     # linear decay to 0
    return 0.0

scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: wsd_lambda(step, 2_000, 95_000, 100_000),
)
```

Peak LR (set on the optimizer) stays fixed; the scheduler's multiplier does the work.

---

## How it works

### The key property: a reusable stable checkpoint

Under cosine, every checkpoint along the curve has a different LR context. If you want to "try a different final phase", you'd need to re-run the entire cosine from scratch — the checkpoint at step 90% of cosine is not equivalent to a freshly-started model because its LR is already low.

Under WSD, the checkpoint at the *end of the stable phase* is a clean starting point for the decay phase. You save it once, then you can:

- Run decay phase with data mix A → model A
- Run decay phase with data mix B → model B
- Run decay phase with mix C and seed 2 → model C

The stable-endpoint model is **fungible with respect to the decay experiment**. This turns mid-training into a cheap iteration loop: you do the expensive Stage 1 once, then explore many Stage 2 variants at a fraction of the cost.

This is how MiniCPM ran dozens of mid-training ablations for the price of one full training run.

### Why stable-LR works

A natural worry: "if LR doesn't decay, won't training just bounce around without converging?" In practice no, for two reasons:

1. **The loss landscape gets flatter as training progresses.** Early on, gradients point in clear directions and the model traverses distance. Later on, the model is in a broad valley where the noise scale of SGD matters more than the LR magnitude. A constant LR that was reasonable early is typically fine (just noisy) later.
2. **The decay phase is what actually "commits" the model.** During stable, the model is exploring a basin. During decay, it settles into the nearest minimum within that basin. The final benchmark number is determined by where you start the decay and what data you see during it — not by whether the stable phase gently annealed.

Put differently: the stable phase builds capability (what the model *can* do); the decay phase locks in behavior (what the model *does*). Cosine conflates these by doing both gradually and simultaneously; WSD separates them.

### Scaling-friendly

Another practical win: under WSD, extending training is trivial. Want 500B more tokens? Just extend the stable phase — no schedule to recompute, no LR plan to re-warp. Under cosine, "add more tokens" means either (a) extending the cosine (changing the effective LR at every step, no longer comparable to the original run) or (b) tacking on a new cosine (a discontinuity in LR). Neither is clean.

MiniCPM uses this to make their "start small, scale up when the recipe works" workflow cheap.

---

## Why it matters

- **Makes mid-training iterable.** The stable-endpoint artifact turns Stage 2 from "one shot, hope it works" into "run 20 variants, pick the best". Arguably the best-kept secret of modern pre-training labs.
- **No LR-shape hyperparameter.** Cosine has an implicit choice (how low does it go?) that matters but is hard to ablate. WSD has a phase-boundary step and a decay length — both unambiguous.
- **Resumable without re-warping.** Extend or truncate the stable phase freely. Useful when you're deciding how much compute to spend as the run progresses.
- **Composes with [model souping](model-souping.md).** Run multiple decay-phase variants from the same stable checkpoint, then soup them. The souping assumption (shared parent) is exactly satisfied.

## Gotchas & tricks

- **Don't reset the optimizer at phase boundaries.** Adam's m, v state carries over from warmup to stable to decay. Resetting optimizer state at the decay boundary throws away useful curvature info and usually costs a few points.
- **Stable LR should be a "peak that you can sustain".** The same LR that is slightly too-high for cosine (causing occasional spikes) is very much too-high for stable, because you sit at that value for weeks. Err on the side of a slightly lower stable LR.
- **Decay length matters.** Too short (< 1% of tokens): model doesn't settle, final loss is noisy. Too long (> 30% of tokens): you're just cosine-decay with extra steps, losing the "re-usable stable endpoint" benefit. 5–15% is the sweet spot most labs land on.
- **Watch for curriculum-style shifts at the decay boundary.** Most WSD deployments swap to a higher-quality data mix exactly when decay starts. If you *don't* swap (i.e. same data into decay as into stable), you'll see smaller benchmark gains — the decay-phase leverage comes from the data, not the LR alone.
- **Not a free lunch for tiny runs.** Under small compute, WSD's stable-then-decay is roughly equivalent to cosine. The "re-fork the endpoint" benefit assumes you actually plan to run multiple decay experiments; if you're running one shot, pick whichever schedule your codebase already has.
- **Stable phase reveals mesa-stability.** If your recipe has an unnoticed slow instability (e.g. attention-logit drift), WSD's long constant-LR phase will surface it before a cosine run would. That's a feature — it makes stability issues visible earlier.

## Sources

- Paper: *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* — OpenBMB / MiniCPM team, 2024 — introduces WSD for LLM pre-training, argues for reusable-stable-endpoint as a core design.
- Paper: *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* — Hägele et al., 2024 — analyzes WSD under compute-optimal scaling, shows the decay phase is what drives the terminal loss improvement.
- Paper: *DeepSeekMath* — DeepSeek, 2024 — uses a multi-step schedule in the WSD family.
