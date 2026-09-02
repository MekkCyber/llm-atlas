# LoRA — Low-Rank Adaptation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Parameter-efficient fine-tuning that freezes the base model and inserts a trainable low-rank update $\Delta W = B A$ (with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$) into selected weight matrices — usually attention $W_Q, W_V$. Cuts trainable parameters by 100–1000× and lets many task-specific adapters share one frozen backbone. At inference, $B A$ can be merged into $W$ for zero overhead.

**Prereqs:** [../../fundamentals/attention.md](../../fundamentals/attention.md)
**Related:** [nora.md](nora.md) · [../../architectures/_normalization.md](../../architectures/_normalization.md)

---

## What it is

Full fine-tuning of a large model updates every parameter and stores a full copy of the trained weights per task — infeasible at scale. LoRA freezes the pretrained $W_0$ and learns only a low-rank residual:

$$
W = W_0 + \Delta W = W_0 + B A
$$

Trainable parameter count is $r(d+k)$ instead of $dk$ — for $d = k = 4096$, $r = 8$: **65,536 params vs 16.7M**, a 256× cut per matrix.

## How it works

**Where.** Applied to attention $W_Q, W_V$ by default in the original paper; modern usage adds $W_K, W_O$, MLP projections, or all linear layers. Not applied to embeddings/norms.

**Initialization.** $A \sim \mathcal{N}(0, \sigma^2)$ (Kaiming-uniform in the reference implementation) and $B = 0$. This makes $\Delta W = 0$ at step 0, so training begins from the exact pretrained function.

**Forward pass.** For input $x$:
$$
h = W_0 x + \frac{\alpha}{r} B A x
$$
The $\alpha / r$ scaling decouples effective learning rate from rank: pick $\alpha = r$ (or $\alpha = 2r$) and rescan LR once, not per rank.

**Training.** Only $A$ and $B$ are optimizer parameters; $W_0$ has no gradient. Memory footprint drops accordingly.

**Deployment.** At inference, either (a) keep the LoRA branch separate (hot-swap adapters per request) or (b) merge $W \leftarrow W_0 + \alpha/r \cdot BA$ for zero overhead.

## Why it matters

- **Enables per-task/per-user adaptation.** Adapter files are ~megabytes; hosting one base + many adapters is cheap.
- **Democratizes fine-tuning.** A 7B model needs ~28 GB of gradient + optimizer state under full FT; LoRA reduces it by ~2 orders of magnitude, enabling single-GPU FT.
- **Composes with quantization.** QLoRA quantizes $W_0$ to 4-bit and trains $A, B$ in higher precision — the standard recipe for consumer-hardware FT.
- **Ubiquitous baseline.** Instruction tuning, RLHF SFT stages, character/persona adapters, style adapters — LoRA is the default until you have a reason not to use it.

## Gotchas & tricks

- **Rank selection.** $r \in \{4, 8, 16, 32\}$ common. Diminishing returns past $r = 16$ for most tasks; large $r$ approaches full FT and loses the "efficient" claim.
- **Which matrices to touch matters more than $r$.** Most-quoted default (Q and V only) is undertrained for many tasks. Applying LoRA to *all* linear layers usually wins if compute allows.
- **Init asymmetry drives early dynamics.** $B = 0$ means early updates flow through the *down-projection* $A$ alone; conditioning of $A$ dominates optimization. See [nora.md](nora.md) for a normalization fix.
- **Learning rate is higher than full FT.** Because effective update is $\alpha/r \cdot BA$, use $10\times$–$100\times$ the LR you'd use for full FT.
- **Merging costs a hair of quality when the base is quantized.** Merging $BA$ back into a 4-bit $W_0$ requires re-quantization, which rounds away part of the adapter. Keep the branch separate for QLoRA deployments.
- **Catastrophic forgetting still happens.** LoRA reduces but does not eliminate distribution shift on the base capability. KL-regularization or reference-model penalties help.
- **Multi-adapter composition.** Simple weighted-average of $BA$ terms works surprisingly often; more principled routing (LoRAHub, S-LoRA) exists but adds complexity.

## Sources

- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., Microsoft, 2021 — arxiv.org/abs/2106.09685.
- Paper: *QLoRA: Efficient Finetuning of Quantized LLMs* — Dettmers et al., 2023 — arxiv.org/abs/2305.14314.
- Code: github.com/microsoft/LoRA, github.com/huggingface/peft.
