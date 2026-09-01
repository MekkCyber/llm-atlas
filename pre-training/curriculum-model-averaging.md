# Curriculum Model Averaging
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Average model checkpoints taken **along a data-curriculum trajectory** — not just late-stage runs from the same seed as in classical [model souping](model-souping.md). Puro-2B uses this as one of its cost levers: a "free quality" step layered on top of any pretraining run that walks a staged curriculum. The averaged checkpoint keeps competence acquired at earlier curriculum stages that a pure end-of-run checkpoint has partly overwritten.

**Prereqs:** [model-souping.md](model-souping.md), [mid-training.md](mid-training.md)
**Related:** [wsd-schedule.md](wsd-schedule.md) · [_lr-schedules.md](_lr-schedules.md)

---

## What it is

Classical model souping averages weights of multiple *sibling* runs — same parent checkpoint, same stage, different seeds or data orderings. Curriculum model averaging averages weights from **different points along the same run's curriculum**: end of Stage 1, end of Stage 2, end of Stage 3, etc. — checkpoints that saw different data mixtures rather than different seeds of the same mixture.

The wins are the same shape (a small but consistent lift over any single checkpoint), but the *why* is different: this variant preserves competence from earlier curriculum stages that later stages have partly overwritten.

## How it works

- **Take checkpoints at meaningful curriculum boundaries** — end of each pretraining stage, or evenly spaced within a WSD-schedule decay phase.
- **Average their state dicts element-wise.** Uniform average is the default; weighted averages are possible when one stage is much more important than others.
- **Evaluate the averaged model.** If the sibling curriculum stages are close enough in weight-space that linear mode connectivity holds, the average sits closer to a shared basin floor than any individual stage.

Works cleanly when the model has stayed inside one loss basin across the curriculum — as tends to happen with WSD-style schedules that anneal the LR to zero at the end of each stage.

## Why it matters

- **Free quality on top of any staged run.** No extra training compute — you're averaging checkpoints you already have.
- **Recovers competence overwritten by later stages** — e.g. broad web knowledge that a late-stage math/code curriculum starts to displace. The averaged model retains both.
- **Composes with sibling souping.** Nothing prevents also averaging over seeds within a stage first, then averaging across stages.

## Gotchas & tricks

- Requires the run to stay inside one basin — a hard mid-training reset (major architecture change, fresh optimizer state, non-connected initialization) breaks linear mode connectivity and the average degrades instead of improving.
- Weighted averages help when stages differ sharply in size or importance; uniform average is the safe default.
- Evaluate the averaged checkpoint against the *best* individual stage, not against Stage 1 — the win is real but small (typically fractions of a benchmark point).

## Sources

- Paper: *Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090* — Luo et al., Tsinghua PACMAN, 2026 — [arxiv](https://arxiv.org/abs/2608.27370)
- Related: *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time* — Wortsman et al., 2022 — the sibling-averaging precedent this technique generalizes.
