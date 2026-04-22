# Pre-Training

*Large-scale unsupervised learning — how base models are trained from scratch on massive corpora, and the parallelism, stability, and infrastructure decisions that make it work.*

---

## What This Is

Pretraining is where raw capability is built. Most of the compute budget, most of the infrastructure complexity, and most of the scaling-law literature live here. This folder covers the math (scaling laws, loss curves), the systems (parallelism strategies, mixed precision), and the operational concerns (stability, loss spikes, recovery) of running a pretraining job.

---

## What Belongs Here

- **Scaling laws** — Kaplan, Chinchilla, compute-optimal ratios.
- **Parallelism** — data parallel, tensor parallel, pipeline parallel, FSDP, sequence parallel.
- **Mixed precision** — FP16, BF16, FP8, loss scaling.
- **Training stability** — warmup, learning rate schedules, gradient clipping, loss spikes.
- **Checkpointing & recovery** — at scale, with sharded state.
- **Pretraining data** — mixture, ordering, curriculum (with [data/](../data/)).
- **Long-context pretraining** — extending sequence length during or after pretraining.

## Reading Order

1. Scaling laws (Kaplan → Chinchilla)
2. Mixed precision training
3. Data parallel & FSDP
4. Tensor & pipeline parallelism
5. Training stability & loss spikes
6. Long-context extension

---

## Overview Pages (taxonomies)

- [LR schedules](_lr-schedules.md) — warmup/cosine/WSD/constant-tail choices.
- [Training stability](_training-stability.md) — loss spikes, normalization tricks, init, clipping.

## Concept Pages (depth)

- [WSD schedule](wsd-schedule.md)
- [Mid-training (Stage 2)](mid-training.md)
- [Model souping](model-souping.md)
- [Multi-Token Prediction (MTP)](mtp.md)
- [FP8 mixed-precision training](fp8-training.md)

---

## Related

- [fundamentals/](../fundamentals/) — optimizers, normalization, initialization.
- [architectures/](../architectures/) — the architecture being pretrained.
- [systems/](../systems/) — distributed infrastructure.
- [data/](../data/) — pretraining corpora and mixtures.
- [quantization/](../quantization/) — the FP8 formats the training uses.
