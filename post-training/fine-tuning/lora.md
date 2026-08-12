# LoRA — Low-Rank Adaptation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Freeze the base model and inject two small matrices `A ∈ R^{d×r}` and `B ∈ R^{r×d}` next to each target weight `W`. The effective update becomes `W + BA` at inference; during training only `A` and `B` learn. With `r ≪ d`, trainable parameters shrink 100–1000× vs full fine-tuning, memory drops proportionally, and multiple task-specific adapters can be swapped or merged at zero inference overhead.

**Prereqs:** [../grpo.md](../grpo.md) (only as a comparison target — LoRA composes with any post-training loss)
**Related:** [mixture-of-lora.md](mixture-of-lora.md), [../_post-training.md](../_post-training.md), [../../architectures/_moe.md](../../architectures/_moe.md)

---

## What it is

Full fine-tuning updates every parameter in `W ∈ R^{d×d}` — expensive in memory (optimizer state ~4× weight size for AdamW) and in storage (one full copy per task). LoRA replaces the full update `ΔW` with a **low-rank factorization**:

```
ΔW = B · A     A ∈ R^{r×d},  B ∈ R^{d×r},  r ≪ d
h = W·x + α/r · B·A·x
```

`A` is initialized Gaussian, `B` at zero (so `ΔW = 0` at step 0). Only `A` and `B` receive gradients; `W` is frozen and its AdamW state is never allocated. `α/r` is a fixed scaling factor — commonly `α = 2r` so the effective step size is stable across `r` choices.

Typical targets: the four attention projection matrices (`W_q, W_k, W_v, W_o`) — sometimes the MLP up/down projections too. Ranks in production: `r ∈ {8, 16, 32, 64}`.

## How it works

- **Training:** forward pass adds the low-rank branch alongside the frozen weight. Backward pass computes gradients only for `A` and `B`. Optimizer state, activations, and gradient buffers all scale with `r·d` instead of `d²`, cutting memory ~10–100×.
- **Inference:** merge `W ← W + (α/r)·B·A` once, ship the merged weights — zero runtime overhead vs the base model. Or keep the adapter separate and switch at request time (LoRA hot-swapping).
- **Composition:** because merged adapters are just additive updates, multiple LoRAs can be summed (with per-adapter scales) to combine skills, though interference grows with the number of concurrently-active adapters.

## Why it matters

LoRA turned per-task fine-tuning from a serving-cost problem into a storage-and-swap problem. A single frozen 70B base can serve dozens of task-specialized personalities by loading a ~50 MB adapter per request instead of a ~140 GB full model per task. It's the underlying mechanism behind Mixture-of-LoRA agent stacks, per-user personalization, and the entire QLoRA-style consumer fine-tuning ecosystem.

The rank-`r` bottleneck is also a useful inductive bias — it implicitly regularizes the update to the low-dimensional subspace where task-specific structure lives, which is empirically close to what full fine-tuning finds anyway.

## Gotchas & tricks

- **Rank vs quality is task-dependent.** `r = 8` is fine for instruction tuning; reasoning-heavy tasks often want `r ≥ 32`. Diminishing returns past `r = 64`.
- **Which layers matter.** Attention `q/v` projections are the highest-ROI targets. Adding MLP LoRAs helps for larger domain shifts.
- **Zero-init `B` is not optional.** If both `A` and `B` are random-init, `ΔW ≠ 0` at step 0 and the base model's behaviour is immediately perturbed — training becomes unstable.
- **α is not a learning rate.** It's a fixed scaling that anchors the effective update magnitude across ranks. Change LR, not `α`, for optimization tuning.
- **Merging kills swapability.** Once you fuse `BA` into `W` for latency, you lose the ability to hot-swap; keep the adapter separate in multi-tenant serving.
- **Doesn't compose with quantized weights automatically.** QLoRA is the specific recipe for training LoRA on top of 4-bit quantized bases with dequantize-on-the-fly linear layers.

## Sources

- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., 2021 — the original method.
- Paper: *QLoRA: Efficient Finetuning of Quantized LLMs* — Dettmers et al., 2023 — LoRA + 4-bit base + paged optimizers.
- Implementation: HuggingFace PEFT library — canonical open-source reference.
