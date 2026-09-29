# Low-Rank Adaptation (LoRA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** LoRA freezes the pretrained weights and learns a rank-$r$ additive update per weight matrix, factorized as $W_{\text{new}} = W_0 + B A$ with $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$. Training memory drops to a small fraction (just $A, B$ and their optimizer state), and at inference the adapter either fuses into $W_0$ (zero-cost) or stays separate for task swapping.

**Prereqs:** [_post-training](../_post-training.md), [attention](../../fundamentals/attention.md)
**Related:** [dpo](../dpo.md), [rejection-sampling](../rejection-sampling.md)

---

## What it is

Fine-tuning a modern LLM by updating every parameter is expensive: optimizer state alone is 2–3× the model size in FP32. LoRA replaces those full-parameter updates with a **learned low-rank correction**:

$$
h = W_0 x + \alpha \cdot B A x
$$

$W_0$ is frozen; only $A$ and $B$ (plus a scalar $\alpha/r$ scaling term) are trained. Rank $r$ is usually 4–64. For a $d \times k$ matrix, LoRA reduces the parameter count of that layer's update from $d \cdot k$ to $r(d + k)$.

Applied per-attention-projection (typically $W_q$, $W_v$, sometimes $W_k$, $W_o$) and optionally per-FFN matrix. At inference, the sum $W_0 + BA$ can be materialized into a single matrix for zero additional latency.

## How it works

- Initialize $A$ with random Gaussian, $B = 0$ so the initial adapter is a no-op.
- Freeze $W_0$; train $A, B$ with a standard downstream objective (SFT loss, DPO, RLVR reward — anything).
- The effective LR on the update is scaled by $\alpha / r$; typical $\alpha \in \{8, 16, 32\}$ regardless of $r$, so an $r=8$ adapter with $\alpha=16$ has a 2× effective LR versus $r=16$ with $\alpha=16$.
- Serving: **fuse** ($W_0 \leftarrow W_0 + BA$) for one task at a time, or **stack** ($A_i, B_i$ per task) to swap adapters at request time.

QLoRA (Dettmers 2023) extends LoRA by keeping $W_0$ frozen in **4-bit NF4** while training $A, B$ in BF16, so a 65B-parameter fine-tune fits on one 48 GB GPU.

## Why it matters

- **Cheap fine-tunes.** Training memory ~ constant in model size (dominated by activations), enabling 70B fine-tunes on a single 8×H100 node.
- **Zero inference overhead** when the adapter is fused.
- **Composable.** Multiple task adapters can be swapped at request time or merged offline; enables serving many specialized fine-tunes from one base.
- **Ecosystem default.** PEFT, TRL, Unsloth, Axolotl, and every HF fine-tune tutorial default to LoRA/QLoRA.

## Gotchas & tricks

- **Rank matters, but so does layer choice.** Applying LoRA only to $W_q, W_v$ underperforms applying it to *all* linear layers when compute allows; that's what modern recipes do.
- **α/r scaling drifts effective LR.** If you sweep $r$, keep $\alpha/r$ fixed rather than $\alpha$.
- **Not every layer needs the same rank.** Recent work (see [lora-merging](lora-merging.md) — not yet written) shows big gains from per-layer rank allocation.
- **Merging LoRAs is not free.** Naive averaging of task adapters causes interference; several methods (TIES, DARE, Net Utility) fix this by keeping only high-utility singular directions.
- **Full fine-tune still wins on some tasks.** For substantial distribution shifts (new language, new modality), LoRA at typical ranks caps out below full FT. Raise $r$ (32–128) or use DoRA / rsLoRA before giving up.

## Sources

- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., Microsoft, 2021 — https://arxiv.org/abs/2106.09685
- Paper: *QLoRA: Efficient Finetuning of Quantized LLMs* — Dettmers et al., 2023 — https://arxiv.org/abs/2305.14314
- Paper: *DoRA: Weight-Decomposed Low-Rank Adaptation* — Liu et al., 2024.
- Paper: *Not All Ranks Are Equal: Budget-Aware LoRA Merging Across Tasks* — Amballa et al., Amazon, 2026 — https://arxiv.org/abs/2609.22237 (motivates per-layer rank allocation).
