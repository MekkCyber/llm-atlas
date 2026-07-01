# Asynchronous Pipeline Parallelism
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Pipeline parallelism synchronously wastes GPU cycles on pipeline bubbles. **Asynchronous** schedules (PipeDream, PipeDream-2BW) eliminate bubbles at the cost of gradient staleness — but staleness was believed to be inherently destabilising and the schedule was largely abandoned at frontier scale. New evidence: the instability was *AdamW-specific*. Modern optimisers like Muon are robust to one-step gradient delay; a lightweight error-feedback correction closes the residual gap for any optimiser, backed by convergence theory.

**Prereqs:** [dualpipe.md](./dualpipe.md), [../pre-training/_training-stability.md](../pre-training/_training-stability.md)
**Related:** [../pre-training/README.md](../pre-training/README.md), [../pre-training/muon-optimizer.md](../pre-training/muon-optimizer.md)

---

## What it is

Pipeline parallelism splits a model along the layer axis across `P` devices. Synchronous schedules (GPipe, 1F1B, DualPipe) trade memory for bubble-freeness. Asynchronous schedules take a different route: never wait — always feed the next micro-batch — and pay for it in **staleness**, where the optimiser sees gradients computed against an older set of weights.

**PipeDream-2BW** is the appealing async schedule: unlike vanilla PipeDream, it guarantees **exactly one step** of gradient delay regardless of pipeline depth. This bounded delay makes theoretical analysis tractable but was thought to still degrade training in practice.

## How it works

**The one-step delay.**
- At learner step `t`, the gradient the optimiser applies was computed against weights `θ_{t-1}`.
- The forward pass happens with `θ_{t-1}`; the backward completes as `θ_t` is being applied; the *next* optimiser step uses that backward result against `θ_t → θ_{t+1}`.
- One-step delay, independent of pipeline depth `P`.

**Why AdamW breaks under one-step delay.**
- AdamW maintains second-moment estimates that assume the gradient distribution is a low-noise function of the current parameters.
- A one-step delay makes gradients a function of *nearby but different* parameters; the second-moment estimate compounds the mismatch across steps, and the effective learning rate drifts.

**Why Muon survives.**
- Muon replaces AdamW's per-parameter second-moment with an orthogonalised update over the weight matrix (via Newton-Schulz iteration).
- The orthogonalisation absorbs the coordinate-wise noise that broke AdamW, without accumulating state that assumes on-policy gradients.
- Empirically robust to one-step delay across scale.

**Error-feedback correction (optimiser-agnostic).**
- Store the gradient error from step `t-1` (the residual not applied due to staleness).
- At step `t`, add the stored residual to the fresh gradient before the optimiser step.
- Theoretically: closes the delay-induced gap for a general class of optimisers, with convergence guarantees for Muon with and without the correction.

**Empirical scope.** Evaluated on models up to **10B parameters**; Muon (± error-feedback) matches synchronous baselines.

## Why it matters

- **Reopens a stranded schedule.** Async pipeline parallel had a decade of pessimistic reputation. Being able to reach it with modern optimisers means real wall-clock savings — pipeline bubbles are one of the last-standing utilisation drains in large runs.
- **Optimiser choice becomes a systems knob.** Which optimiser to use is now partly determined by which pipeline schedule you want to run.
- **Complementary to DualPipe.** DualPipe pays with 2× parameter memory; async pipeline pays with a Muon-shaped optimiser. Different tradeoff surface.

## Gotchas & tricks

- **Only *one-step* delay is theoretically covered.** Longer delays (vanilla PipeDream, `P > 2` steps) are still hard; the paper's guarantees don't extend.
- **Muon has its own tuning surface.** Newton-Schulz step count, learning-rate scaling, and warmup differ from AdamW; expect a re-tuning phase.
- **Error-feedback storage costs a gradient's worth of memory** — non-trivial at 10B+.
- **Not a free swap over DualPipe.** For MoE-heavy models with cross-node all-to-all, DualPipe's comm-overlap advantages may still dominate.

## Sources

- Paper: *One-Step Gradient Delay Is Not a Barrier for Large-Scale Asynchronous Pipeline Parallel LLM Pretraining* — Zmushko et al. (Yandex / ISTA), 2026 — [arXiv:2606.30634](https://arxiv.org/abs/2606.30634).
- Paper: *PipeDream-2BW: Memory-Efficient Pipeline-Parallel DNN Training* — Narayanan et al., 2020.
