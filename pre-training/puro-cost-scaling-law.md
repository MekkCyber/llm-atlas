# Puro Cost Scaling Law
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **cost-side** scaling law fit across the Puro-2B checkpoint collection: expected average model performance as a function of **dollars of pretraining cost**, with hardware and precision held fixed. Complements classical Chinchilla-style *compute-optimal* laws (parameters vs tokens) by answering the operational question: *given a dollar budget, what performance can I reach?*

**Prereqs:** [fp8-training.md](fp8-training.md), [wsd-schedule.md](wsd-schedule.md)
**Related:** [model-souping.md](model-souping.md) · [curriculum-model-averaging.md](curriculum-model-averaging.md)

---

## What it is

Chinchilla-style laws relate loss to parameters and tokens, assuming compute is fungible across hardware. In practice a small lab's compute isn't fungible — RTX 5090 hours are cheap, H100 hours are not, and precision (FP8 vs bf16 vs FP32) changes throughput per dollar. The Puro Cost Scaling Law fits performance directly to dollar cost within a fixed hardware+precision regime, which is what a small lab planning a run actually needs.

## How it works

The Puro-2B collection spans a grid of **token budgets × recipe variants**. For each checkpoint:

- **Cost** is known — hours on RTX 5090s at market rates, given the chosen precision, optimizer, and data-loader efficiency.
- **Performance** is the average across an eval suite.

The paper fits a curve of *expected average performance vs cost* to this grid. Under the fit, **~$4.4K** is sufficient to reach Qwen2-1.5B performance under the paper's eval protocol — this is the sharp prediction the report is built around, and the reason the branding lands on "$5090" (RTX 5090 + a $5,090 training budget cap for the 1.5B-parameter class).

## Why it matters

- **Operational scaling law for small labs.** Answers "given a dollar budget, how good a model can I train?" rather than "given a compute budget in FLOPs, what's the compute-optimal (N, D)?"
- **Puts hardware + precision inside the law** rather than outside it. Two runs at the same FLOP count on H100/bf16 and RTX 5090/FP8 have different dollar costs and, per this law, different reachable frontiers.
- **Portable methodology, not portable coefficients.** The fit is specific to Puro-2B's recipe on RTX 5090s at market rates. The *methodology* — grid over budgets × variants, fit performance vs cost — is what other labs should copy.

## Gotchas & tricks

- The fit's coefficients don't transfer across hardware, precision, or optimizer choices. Re-fit for your regime.
- Market rates for consumer GPUs are volatile; dollar predictions have to be re-anchored when rental prices shift.
- Does not replace Chinchilla-style compute-optimality — for choosing (N, D) at fixed FLOPs a compute-optimal law is still the right tool. Puro Cost Scaling Law is downstream of that choice.
- The eval suite matters. A cost law fit against a broad average is different from one fit against a specific benchmark; publish which suite the fit uses.

## Sources

- Paper: *Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090* — Luo et al., Tsinghua PACMAN, 2026 — [arxiv](https://arxiv.org/abs/2608.27370)
- Related: *Training Compute-Optimal Large Language Models* — Hoffmann et al., 2022 — the Chinchilla law this complements.
