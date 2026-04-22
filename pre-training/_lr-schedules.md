# LR Schedules

*Taxonomy — the shape of the learning-rate-over-time function during pretraining.*

**TL;DR:** The LR schedule is the multiplier on peak LR across training steps: warmup → main body → [optional final decay]. Modern LLMs use one of three shapes: **classical cosine** (peak → low, one curve), **two-stage cosine** (cosine to a plateau, then further anneal during [mid-training](mid-training.md)), or **[WSD](wsd-schedule.md)** (warmup, stable at peak, then sharp decay). The main design question is where the aggressive decay happens and what data the model sees during it.

**Related taxonomies:** [_training-stability](_training-stability.md)
**Depth files covered here:** [wsd-schedule](wsd-schedule.md) · [mid-training](mid-training.md)

---

## The problem

Gradient-descent-based optimizers need a learning rate. A single fixed LR fails: too low and training is slow; too high and training diverges. Empirically, optimal LR is a function of step:

- **Early in training** — gradients are large and models are far from any minimum. A small LR avoids catastrophic first steps.
- **Mid training** — model is in a useful part of weight space; a large LR accelerates progress.
- **Late in training** — model is near a minimum; large LR bounces out of the basin, small LR settles in.

The schedule is the mapping `step → LR multiplier`. Getting the shape right matters as much for final quality as most architectural choices.

## The shared pattern

Every LR schedule decomposes into three phases that may or may not be distinct:

1. **Warmup.** Linear ramp from ~0 to peak LR over the first ~0.5–2% of training. Every modern schedule has this. Its job is to let Adam's running statistics stabilize before large updates land.
2. **Main body.** The bulk of training. Depending on schedule, LR can be decaying continuously (cosine), pinned constant (WSD's stable), or slowly decaying to a plateau.
3. **Final decay.** The last 5–20% of training where LR anneals toward zero. This is where model behavior "commits" — the data seen here disproportionately shapes final weights.

The difference between schedules is how aggressive phase 2's decay is and whether phase 3 is clearly distinct or just the tail of phase 2.

## Variants

| Technique | Shape | Phase-2 behavior | Main tradeoff | When it wins |
| --- | --- | --- | --- | --- |
| Classical cosine | Warmup → single cosine → near-zero | Continuous gradual decay across whole run | No distinct "commit" phase; data late in run has the most leverage on irreversible weight changes | Simple, canonical; older models (GPT-3, early LLaMA) |
| **Two-stage cosine** | Warmup → cosine to plateau (Stage 1) → further anneal during [mid-training](mid-training.md) (Stage 2) | Cosine decays to ~30% of peak, not zero | Boundary between stages must be planned in advance | **[OLMo 2](../case-studies/olmo-2.md), Llama 3** |
| [WSD](wsd-schedule.md) | Warmup → constant at peak → sharp decay | LR stays at peak until decay phase starts | Stable-endpoint checkpoint is reusable; lets you fork many decay experiments from one Stage 1 | MiniCPM, DeepSeekMath — when you want to iterate on Stage 2 data mixes cheaply |
| Inverse-square-root | Warmup → `1/√step` | Smooth decay without a pre-set duration | Hard to compare runs at fixed token counts | T5-era, older |
| Linear decay | Warmup → linear ramp down | Simple | Typically worse than cosine at fixed compute | Small runs where simplicity matters |
| Cosine with restarts (SGDR) | Warmup → cosine → reset → cosine → … | Multiple minima with LR resets | Rare in LLMs | Computer vision; occasional transfer to LLMs |

## How to choose

**For a modern frontier-scale pretraining run, pick between two-stage cosine and WSD.**

- **Two-stage cosine** is the safe default and what Llama 3 and OLMo 2 use. Phase 1's cosine decays to a non-zero plateau; Phase 2 ([mid-training](mid-training.md)) anneals further to near-zero on a curated data mix. Continuous LR curve, clean.
- **WSD** is the iteration-friendly choice. Phase 1's LR pinned at peak means the Stage-1 checkpoint is fungible across different decay experiments — you can run many Stage-2 variants from one stable endpoint and compare cheaply. Excellent when the Stage-2 data mix is itself under active research.

**Classical single-stage cosine** is what older papers used. The problem: your LR anneals to near-zero while still on generic Stage-1 data, which means you spend the most benchmark-relevant fraction of your compute on your lowest-quality data. Modern labs moved off it for exactly this reason.

**Linear decay, inverse-sqrt, cosine-with-restarts** — skip for pretraining at scale. Use them for small experiments or specific non-LLM cases.

### Rule of thumb: match the schedule to the data curriculum

If your data mix is flat (same distribution throughout training), classical cosine is fine. If you have a two-phase curriculum (bulk web → curated late mix), use two-stage cosine or WSD so the aggressive LR decay aligns with the high-quality data. The schedule and the curriculum decisions are linked.

## Adjacent but distinct

- **Optimizer choice** (Adam, AdamW, Lion, Muon). Separate axis — pick optimizer and schedule independently.
- **Gradient clipping norm.** Stability knob, not a schedule.
- **Batch-size schedules.** Separate topic; sometimes co-designed with LR (linear scaling rule) but orthogonal.
- **Weight-decay schedules.** Some labs decay weight decay alongside LR. Worth naming but tangential.

## Sources

- Paper: *Attention Is All You Need* — Vaswani et al., 2017 — inverse-sqrt original.
- Paper: *SGDR: Stochastic Gradient Descent with Warm Restarts* — Loshchilov & Hutter, 2017 — cosine and restarts.
- Paper: *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* — OpenBMB, 2024 — introduces WSD for LLM pretraining.
- Paper: *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* — Hägele et al., 2024 — WSD scaling-law analysis.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — two-stage cosine schedule in detail.
- Paper: *2 OLMo 2 Furious* — AI2, 2024 — two-stage cosine with explicit [mid-training](mid-training.md) phase.
