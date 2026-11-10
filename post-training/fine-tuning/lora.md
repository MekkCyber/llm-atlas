# Low-Rank Adaptation (LoRA)
*Depth — parameter-efficient fine-tuning by training only a low-rank update to each weight matrix.*

**TL;DR:** For each weight matrix `W ∈ R^(d_out × d_in)` you want to fine-tune, freeze `W` and learn two small matrices `A ∈ R^(r × d_in)`, `B ∈ R^(d_out × r)` with `r ≪ min(d_in, d_out)`. The fine-tuned weight is `W + BA · (α/r)`. Trainable parameters drop by orders of magnitude (often 100–1000×), there is zero inference overhead after merging, and multiple adapters can be hot-swapped at serve time.

**Prereqs:** [_post-training](../_post-training.md), [README (fine-tuning)](README.md)
**Related:** [dpo](../dpo.md), [grpo](../grpo.md), [multi-objective-rl](../multi-objective-rl.md), [on-policy-distillation](on-policy-distillation.md)

---

## What it is

Full fine-tuning updates every parameter of a large pretrained model. For a 70B model, that is ~140 GB of optimizer state (AdamW: 2× params in fp32), a very expensive proposition for a specialization task. The LoRA observation: **the update that fine-tuning finds has low intrinsic rank** — the effective change `ΔW = W_tuned − W_pretrained` has a small singular-value tail. So you can parameterize `ΔW` directly as a low-rank factorization and train only that factorization.

```
Forward with LoRA:
    h = Wx + (α/r) · B(Ax)

Trainable: A, B    (r · (d_out + d_in) parameters)
Frozen:    W       (d_out · d_in parameters)
```

At inference, `W + (α/r)·BA` can be *merged* back into a single matrix with zero runtime cost. Before merging, LoRA adds one small matmul per adapted layer.

## How it works

### The factorization

For each targeted linear projection:

- `A ∈ R^(r × d_in)` — init to Gaussian noise.
- `B ∈ R^(d_out × r)` — init to zero.

With `B = 0` at start, the model behaves exactly like the base — training only *adds* a learned low-rank perturbation. `A` and `B` are trained with standard optimizers; everything else stays frozen.

### The scaling factor α/r

`α` is a fixed hyperparameter; `α/r` controls the magnitude of the LoRA contribution relative to the base weight. Common recipes fix `α = r` (ratio 1), or `α = 2r` (ratio 2) for stronger adaptation. Tuning `α` is roughly equivalent to tuning the LoRA learning rate — one or the other.

### Which modules to adapt

The original paper adapts `W_q` and `W_v` in attention. Modern recipes typically adapt more:

- All four attention projections (`W_q, W_k, W_v, W_o`).
- The MLP up- and down-projections.
- Sometimes the embedding and output projections.

Wider coverage improves quality at small `r`; narrow coverage suffices for close-to-base tasks (style tuning, instruction following).

### Rank selection

Typical `r` values: 4, 8, 16, 32, 64. Higher-rank LoRA is more expressive but more expensive and prone to overfitting. For instruction-tuning a 7B model, `r = 16` is a strong default. For aggressive capability change (new domain, long reasoning), `r = 64` or higher is needed.

### Serving multiple adapters

Because `BA` is tiny per adapter, a single base model can hold hundreds of LoRA adapters in memory and switch between them cheaply. This is the foundation of multi-tenant serving (one base, many personas or tasks) and of multi-objective methods like [LambdaMix](../multi-objective-rl.md) that train one adapter per objective and mix them at inference.

## Why it matters

- **Opens fine-tuning to consumer hardware.** A 70B base model with `r = 16` LoRA has ~40M trainable params — trainable on a single A100 or even a 4090 with gradient checkpointing.
- **Zero inference overhead.** After merging, a LoRA-tuned model is indistinguishable from a full fine-tune at runtime.
- **Composable.** Multiple adapters can be summed, interpolated, or routed to. Enables parameter-efficient multi-task, multi-objective, and multi-user setups.
- **The default PEFT method.** Nearly every open instruction-tuning recipe (Alpaca, Zephyr, Vicuna, Tülu 3) uses LoRA for budget and ablation scaling. Full fine-tuning is used for frontier runs; LoRA for everything else.

## Gotchas & tricks

- **`B = 0` init is non-negotiable.** Initializing `B` nonzero means the model diverges from the base at step 0 — lost ground before learning anything. The trick of "start as base, drift toward tuned" is the whole point.
- **QLoRA stacks on quantization.** Freeze `W` in 4-bit (NF4) while keeping LoRA `A, B` in bf16. Reduces memory another 4× with ~no quality loss. [QLoRA paper, Dettmers 2023.]
- **DoRA decouples magnitude and direction.** Each weight is split into magnitude `m` and direction `W/‖W‖`; direction gets a LoRA update, magnitude gets a scalar update. Closes some of the quality gap with full fine-tuning at the same rank.
- **Rank doesn't need to be uniform.** Per-layer or per-head rank (AdaLoRA) can allocate budget where gradients are largest.
- **Merging can degrade numerics.** `W + (α/r)·BA` in low precision can lose precision versus keeping them separate. For maximum quality, serve LoRA unmerged (one extra matmul per layer); for maximum throughput, merge.
- **LoRA ≠ SFT replacement for capability change.** Teaching a model a genuinely new capability (new language, new reasoning depth) often needs higher ranks and more modules adapted than a stylistic fine-tune. Don't assume `r = 16` on attention-only is enough for every task.
- **LoRA on preference training works.** DPO-LoRA and GRPO-LoRA both train well; in fact most open DPO recipes default to LoRA because it's cheap enough to run sweeps.

## Sources

- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* — Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen, 2021 — arXiv 2106.09685.
- Paper: *QLoRA: Efficient Finetuning of Quantized LLMs* — Dettmers et al., 2023 — 4-bit base + LoRA.
- Paper: *DoRA: Weight-Decomposed Low-Rank Adaptation* — Liu et al., 2024.
- Paper: *Steerable Multi-Objective GRPO with Jointly Trained LoRA Adapters (LambdaMix)* — Karouzos & Aletras, 2026 — arXiv 2611.09634 — adapter-per-objective RL.
