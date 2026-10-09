# Recurrent-State Quantisation (STEPQuant)

*Depth — mixed-precision quantisation of the per-request recurrent state in linear-attention / delta-rule models.*

**TL;DR:** Linear-attention and delta-rule models (Mamba, RWKV, Kimi-Linear) carry a **fixed-size recurrent state** per request. At serving scale, this state dominates memory much like the KV cache does for Transformers. Naïve PTQ on the state hurts accuracy because quantisation errors travel forward through every state update. STEPQuant allocates bit precision per-entry based on **lifetime** (how many future updates the error affects) and **location** (key-rows vs. value-columns differ in magnitude and sensitivity), with separate row and column scales.

**Prereqs:** [fp8](fp8.md), [_number-formats](_number-formats.md)
**Related:** [mla](../architectures/mla.md)

---

## What it is

For a linear-attention model, the recurrent state $S_t \in \mathbb{R}^{d_k \times d_v}$ evolves by a delta-rule update:

$$
S_t = \alpha_t \cdot S_{t-1} + \beta_t \cdot k_t v_t^\top
$$

The state is kept per request, so with $N$ concurrent requests the serving system holds $N$ copies. Compressing $S$ to lower precision is the direct analogue of INT4/INT8 KV-cache compression for Transformers.

STEPQuant quantises $S$ with a mixed-precision scheme whose budget is allocated along two axes:

1. **Error lifetime** — entries that persist longer across future updates get more bits.
2. **Error location** — key-rows and value-columns have different sensitivities; separate scales per axis.

## How it works

### Error propagation analysis

A quantisation error $\epsilon$ introduced at step $t$ propagates into $S_{t+1}, S_{t+2}, \ldots$ scaled by the running product of the $\alpha$ terms. The paper tracks:

- **Lifetime proxy:** $\prod_{s=t}^{T} \alpha_s$ — how fast the error decays.
- **Spatial impact:** magnitude of the error projected into the key-row and value-column directions, measured on calibration data.

Entries with long lifetime × high spatial impact get the most bits; short-lifetime, low-impact entries can be driven to 4 bits or lower.

### Row/column scales

Rather than a single scalar scale for the whole state, STEPQuant fits:

- A per-row scale on the key dimension $d_k$.
- A per-column scale on the value dimension $d_v$.

Both are cheap to store (vectors, not matrices) and recover most of the FP32 range even at low bit widths.

### Integration

Implemented in SGLang. Only the recurrent state is quantised; weights and activations keep their original precision.

## Why it matters

Linear-attention models were pitched as the "no KV cache" alternative to Transformers. STEPQuant shows the recurrent state is **their** memory wall — but also that it's at least as compressible.

Measured results:

- Near-FP32 accuracy at a nominal **6-bit** budget on Qwen3.8-27B and Kimi-Linear-48B-A3B-Instruct.
- **4-bit** STEPQuant beats uniform **INT8**.
- Over **5× state compression**; up to **68.7% lower total serving memory**.

## Gotchas & tricks

- **Calibration data matters.** The spatial-impact estimate uses forward passes on representative prompts; mismatched calibration degrades low-bit assignments.
- **Doesn't help short-context single-request workloads.** The win is at serving scale with many concurrent requests, where state × batch dominates.
- **Separate from KV-cache quantisation.** This is a different axis — Transformers and linear-attention models have different memory bottlenecks, and the techniques don't substitute for each other.
- **Not a weight / activation quantiser.** Compose with existing weight-quant stacks (GPTQ, AWQ) for full-stack compression.

## Sources

- *When and Where Errors Matter in Delta-Rule Recurrent State Quantization* — Xu, Lin, Wu, Guo, Zhang, Lu, Sun, Wei — 2026 — [arXiv:2609.38169](https://arxiv.org/abs/2609.38169)
