# Looped Transformers
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A looped Transformer wraps a small block (or the whole trunk) of layers and re-executes it $k$ times per token, where $k$ can vary per token or per input. Two payoffs: (1) **weight sharing** lets a shallow-parameter model behave like a much deeper one, and (2) **depth-adaptive inference** — spend more loops on hard tokens and fewer on easy ones. Recent instances: Ouro-1.4B, Huginn-3.5B, plus the older Universal Transformer and Albert lineage.

**Prereqs:** [transformer-block](./transformer-block.md), [attention](../fundamentals/attention.md)
**Related:** [mtp](../pre-training/mtp.md), [continuous-depth-batching](../inference/continuous-depth-batching.md)

---

## What it is

A standard Transformer stacks $L$ distinct blocks; each block has its own parameters. A **looped Transformer** shares parameters across depth: the same block runs $k$ times in sequence, forming an effective depth of $k$ while paying the memory of only one block's weights.

Two axes:

- **What is shared.** Fully-looped ("all layers are one shared block, run $k$ times") vs partially-looped ("N shared blocks, each run $k$ times, plus a few unique outer layers"). Fully-looped is the cleanest for depth-adaptive inference.
- **How $k$ is chosen.** Fixed at inference (same $k$ for every input), input-adaptive (predicted per input), or **token-adaptive** (each token exits the loop when a halting signal fires; Universal Transformer style with Adaptive Computation Time).

## How it works

- **Training.** Loop the block for a scheduled $k$ during forward; backprop through the unrolled graph. Some designs (e.g. Ouro) train with a mix of $k$ values so the model learns to be useful at multiple depths; others fix $k$ at train time and interpolate at inference.
- **Halting head.** For token-adaptive $k$, add a small scalar head on the block's output that predicts "am I done?"; loop until the halting probability crosses a threshold (or a hard maximum).
- **KV cache.** Each loop step produces a new KV state; the total cache per token is $k$ × per-layer cache. Standard batchers assume a fixed depth and cannot easily handle mixed-$k$ tokens in the same batch — motivating [continuous-depth-batching](../inference/continuous-depth-batching.md).

## Why it matters

- **Depth-adaptive compute per token.** The prevailing hypothesis for how frontier models could allocate more thought to hard tokens without a full CoT chain.
- **Shallow-parameter, deep-effective models.** A 3.5B looped LM can match a much larger fixed-depth model on the tasks that benefit from more thinking, at a fraction of the weight budget — good for on-device settings.
- **Composes with early-exit** (a special case of $k = 1$ for confident tokens) and with **speculative decoding** (loop the draft model longer when the verifier disagrees).

## Gotchas & tricks

- **Serving efficiency is the blocker.** Without depth-aware batching, mixed-$k$ tokens force decode workers to run at the maximum $k$ in the batch — you pay the worst-case cost regardless.
- **Halting-head calibration.** Poorly calibrated halting collapses either to "always loop max" (no savings) or "always exit early" (quality loss). Temperature scaling and per-layer thresholds help.
- **Non-looped side layers dominate.** Big embeddings or unshared final blocks eat most of the wall-clock and shrink adaptive-inference wins; fully-looped architectures serve best.
- **Training instability at large $k$.** Very deep unrolls have exploding activations; use QK-norm and reordered-norm patterns, or gradient-checkpoint every few loops.
- **Not the same as recurrent attention (Mamba, RWKV).** Looped Transformers keep global attention every loop; state-space and linear-attention models replace attention with a recurrent kernel. Different family.

## Sources

- Paper: *Universal Transformers* — Dehghani et al., DeepMind, 2018 — the early depth-adaptive Transformer.
- Paper: *ALBERT* — Lan et al., 2019 — parameter sharing across layers as a compression technique.
- Model: *Ouro-1.4B* — early 2026 open-source fully-looped LM.
- Model: *Huginn-3.5B* — 2026 partially-looped LM.
- Paper: *Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching* — Schwethelm, Rueckert, Kaissis — TUM + Helmholtz Munich, 2026 — https://arxiv.org/abs/2608.09444
