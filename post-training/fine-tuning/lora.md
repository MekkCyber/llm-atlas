# LoRA (Low-Rank Adaptation)
*Depth — adapter-based fine-tuning that trains two tiny matrices per layer instead of the full weight matrix.*

**TL;DR:** For each targeted linear layer with weight $W \in \mathbb{R}^{d \times k}$, LoRA freezes $W$ and learns a low-rank residual $\Delta W = B \cdot A$ with $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, $r \ll \min(d, k)$. Only $A, B$ are updated; forward pass becomes $y = (W + \alpha / r \cdot B A) \cdot x$. Trainable parameter count drops by 100×–1000×, and the adapter can be *merged* back into $W$ at inference time with zero overhead. Introduced in Hu et al., 2021; now the default parameter-efficient fine-tuning (PEFT) method for LLMs.

**Prereqs:** [README.md](./README.md), [../_post-training.md](../_post-training.md)
**Related:** [../../architectures/multi-head-attention.md](../../architectures/multi-head-attention.md)

---

## What it is

Full fine-tuning updates every parameter — gigabytes of state per task, slow, and risks catastrophic forgetting. **LoRA** parameterizes the fine-tune delta as a product of two small matrices:

$$
W' = W + \Delta W = W + \frac{\alpha}{r} B A
$$

with $A$ zero-initialized or Gaussian-initialized and $B$ initialized to zero (so $\Delta W = 0$ at the start — the fine-tune begins from the frozen behavior).

The rank $r$ is tiny — typically 4, 8, 16 — and $\alpha$ is a scaling constant (often $\alpha = r$ or $2r$). Trainable parameter count per adapted layer: $r(d + k)$, versus $d \cdot k$ for full fine-tuning.

## How it works

1. **Pick target layers.** Classic choices: `W_q` and `W_v` in attention. Modern practice: all linear layers (`q`, `k`, `v`, `o`, and MLP up/gate/down).
2. **Initialize $A$ Gaussian-small, $B$ zero.** Zero init of $B$ means the adapter is identity at step 0.
3. **Forward.** Compute $y = W x + (\alpha / r) \cdot B A x$. The two paths run in parallel at train time.
4. **Backward.** Only $A, B$ receive gradients; the frozen $W$ saves optimizer-state memory (no Adam moments, no gradient tensor).
5. **Merging.** At inference, compute $W_{\text{merged}} = W + (\alpha/r) \cdot B A$ once; serve $W_{\text{merged}}$ as a drop-in replacement. Zero inference overhead.

## Why it matters

- **100×–1000× fewer trainable parameters.** A 70B model with full fine-tuning requires ~1 TB of Adam optimizer state; a LoRA at $r=16$ on all linear layers trains in a few GB. Fits on a single H100.
- **Composable adapters.** Train one LoRA per task; load the right one at serve time. One base model → many fine-tuned behaviors.
- **No inference cost after merging.** Unlike adapter layers or prompt-tuning, a merged LoRA weighs the same as the base.
- **Underpins modern alignment stacks.** DPO, instruction tuning, RLHF — many open recipes use LoRA to stay in memory budget.
- **Preserves base capabilities.** Full fine-tuning overwrites pretrained knowledge; LoRA at low rank constrains the delta geometrically, mitigating forgetting.

## Gotchas & tricks

- **Rank $r$ sensitivity.** $r=4$–$16$ handles most instruction tuning; reasoning or long-horizon tasks benefit from $r=32$–$64$. Too small ⇒ underfits; too large ⇒ no reason to use LoRA.
- **$\alpha/r$ scaling.** Fixed $\alpha$ with varying $r$ changes the learning rate implicitly. Papers disagree; the safe convention is $\alpha = r$ (or scale the LR instead).
- **Layer coverage.** Attention-only LoRA under-trains MLPs; "all-linear" LoRA is now standard.
- **Merging mid-training.** If you want to continue training *after* merging, re-initialize a fresh LoRA on the merged weights.
- **QLoRA for quantized bases.** When the base is 4-bit quantized, LoRA is applied in fp16 on top; the memory win stacks. See Dettmers et al. 2023.
- **Not for architectural changes.** LoRA tweaks weights; it cannot change layer counts, add heads, or extend context length. For those, do a short full-parameter continued pretraining.
- **Catastrophic forgetting is reduced, not eliminated.** Stacking many task-specific LoRAs sequentially still drifts. For long task sequences, use adapter-mixture methods.

## Sources

- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., Microsoft, 2021 — the original method.
- Paper: *QLoRA: Efficient Finetuning of Quantized LLMs* — Dettmers et al., 2023 — LoRA on top of 4-bit NF4 bases.
- Paper: *The Expressive Power of Low-Rank Adaptation* — Zeng & Lee, 2023 — theoretical analysis of why low rank suffices.
- Paper: *Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It* — Jin, Deng, Wang, 2026 — [arXiv:2609.36585](https://arxiv.org/abs/2609.36585) — early-layer rank-8 LoRA as a depth-use intervention.
