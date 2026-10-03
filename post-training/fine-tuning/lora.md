# LoRA — Low-Rank Adaptation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Fine-tune a frozen pretrained model by learning a *low-rank update* $\Delta W = BA$ (with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$) added to selected weight matrices. Trainable parameters shrink by 100×–1000× vs full fine-tuning with negligible quality loss on most adaptation tasks. The default parameter-efficient fine-tuning (PEFT) method and the base for QLoRA, DoRA, VeRA, and many variants.

**Prereqs:** [../_post-training.md](../_post-training.md), [../README.md](../README.md)
**Related:** [../rejection-sampling.md](../rejection-sampling.md) · [../../interpretability/README.md](../../interpretability/README.md)

---

## What it is

Full fine-tuning updates every parameter of a large pretrained model, which is memory-heavy (optimizer state ≈ 2–4× param count), slow, and produces a full-size checkpoint per task. LoRA (Hu et al., 2021) replaces the gradient update on selected weight matrices with a factored low-rank update:

$$
W_{\text{eff}} = W_{\text{frozen}} + \Delta W, \qquad \Delta W = B A, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k}
$$

Only $A$ and $B$ receive gradients. Rank $r$ is typically 4, 8, 16, 32, or 64. For a $d{=}4096$-dim projection, $r{=}8$ cuts trainable params from $16\mathrm{M}$ to $65\mathrm{k}$ (≈250×).

Usually applied to the attention projection matrices ($W_q$, $W_k$, $W_v$, $W_o$) and sometimes MLP projections; keeping normalisation and embeddings frozen.

## How it works

- Initialize $A \sim \mathcal{N}(0, \sigma^2)$ and $B = 0$ so $\Delta W = 0$ at the start — training begins from the pretrained point.
- Forward: $y = (W_{\text{frozen}} + \alpha / r \cdot BA) x$. The $\alpha/r$ scalar is a user-set scaling; common defaults $\alpha = r$ or $\alpha = 2r$.
- Backward: only $A$ and $B$ are optimized. $W_{\text{frozen}}$ receives no gradient and stores no optimizer state.
- Merge at inference: compute $BA$ once and add to $W_{\text{frozen}}$. The result has the same shape as the original weight — zero inference-time overhead.

Variants:
- **QLoRA** quantizes the frozen base to 4-bit NF4 while LoRA adapters train in 16-bit. Enables fine-tuning 65B models on a single GPU.
- **DoRA** decomposes the weight update into magnitude and direction, applying LoRA to the direction only. Improves quality at the same rank.
- **VeRA, LoRA-FA** share random matrices across layers for extreme parameter efficiency.

## Why it matters

- **Memory.** Optimizer state scales with trainable params, not model size; $r{=}8$ LoRA on a 70B model fits where full FT needs 500+ GB of optimizer state.
- **Multi-task adapters.** You can store hundreds of task-specific $BA$ pairs and hot-swap them at inference without reloading the base.
- **Surprising capability unlock.** A 2026 result shows a rank-8 LoRA at a *single early layer* unlocks reasoning depth frozen base weights wouldn't use: Qwen3-8B goes from 15.5% → 99% on 24-line program chain-following; interpretability analysis shows the LoRA starts an attention relay through middle layers. See [../../interpretability/relay-circuits.md](../../interpretability/relay-circuits.md) once added.
- **Workhorse of the ecosystem.** Hugging Face PEFT, diffusers' LoRA infrastructure, and most open fine-tunes ship as LoRA adapters.

## Gotchas & tricks

- **Rank bites back on long training.** $r$ too small saturates performance; $r$ too large approaches full FT cost without the quality lift. Sweep $r \in \{8, 16, 32, 64\}$; most tasks peak at $r \le 32$.
- **The $\alpha$ scaling isn't a free parameter.** Doubling $\alpha$ at fixed $r$ amplifies the update and often requires an LR drop. Rule of thumb: tune LR, then set $\alpha$ to a defensible default (often $\alpha = r$).
- **Which matrices.** Attention-only LoRA is the common starting point; adding MLP projections helps on hard tasks at ~2× the param cost. Embedding/LM-head LoRA is unusual.
- **Dropout on adapters** is cheap insurance against overfitting on small datasets; LoRA + dropout is the TRL/transformers default.
- **Merging multiple adapters.** Simple weighted sums (A·LoRA + B·LoRA) usually hurts; dedicated routing (LoRA-Hub, X-LoRA, S-LoRA at inference) is the current state of the art.

## Sources

- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* — Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen; Microsoft, 2021 — the original method.
- Paper: *QLoRA: Efficient Finetuning of Quantized LLMs* — Dettmers et al., 2023.
- Paper: *DoRA: Weight-Decomposed Low-Rank Adaptation* — Liu et al., 2024.
- Paper: *Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It* — Jin, Deng, Wang, 2026 — [arXiv:2609.36585](https://arxiv.org/abs/2609.36585). Single-layer rank-8 LoRA as an interpretability-informed depth unlock.
