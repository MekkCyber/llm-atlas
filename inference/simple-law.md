# SIMPLE Law — Analytical Energy/Cost Model for LLM Inference
*Depth — a four-parameter decomposition that predicts energy and runtime without per-model fitting.*

**TL;DR:** A compact analytical cost model for LLM inference that separates **prefill** from **decode**, and inside decode separates **KV-cache movement** from **next-token compute**. Four physically motivated parameters, fit once per model family (not per model). Across **13 dense LLMs from 6 families on H100 GPUs**: **R² = 0.969**, **11.97% MAPE** — **3.1–3.5× lower error than prior models** without per-model recalibration.

**Prereqs:** [README.md](./README.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md) · [../quantization/README.md](../quantization/README.md)

---

## What it is

LLM serving cost has mostly been modeled as a black-box regression over model size, batch, and context length, calibrated per model. The SIMPLE law factors the cost into four **physical components** that correspond to the actual kernels:

1. **Prefill compute** — matmuls over the full prompt. Quadratic in prompt length for attention, linear for FFN.
2. **Decode compute per new token** — FFN + attention over the KV cache. Depends on model size and precision.
3. **KV cache read** — memory-bandwidth-bound cost of reading the cache for each new token. Linear in context length.
4. **Fixed per-request overhead** — scheduling, communication, collectives.

Each of these has a parameter; the parameters are physically interpretable (FLOPs/tok, bytes/tok, const/request).

## How it works

Given a request with prompt length $P$ and $G$ generated tokens, total time (or energy, same shape) is:

$$
E \;\approx\; \underbrace{(\alpha_{\text{p}} P + \beta_{\text{p}} P^2)}_{\text{prefill}} \;+\; G \cdot \underbrace{(\alpha_{\text{d}} + \gamma \,(P + G/2))}_{\text{per-decode-step}} \;+\; \delta
$$

- $\alpha_{\text{p}}, \beta_{\text{p}}$ — prefill FFN and attention coefficients.
- $\alpha_{\text{d}}$ — per-token decode compute.
- $\gamma$ — per-token KV read cost (scales with average context length during decode).
- $\delta$ — fixed per-request overhead.

Fit once per model family (e.g. Llama 3 / Qwen 2.5) on a handful of calibration traces; generalizes across sizes in that family. Per-model calibration is unnecessary because the physical constants are shared.

## Why it matters

Energy / cost modeling drives capacity planning, architecture choices, and quantization tradeoffs. A compact analytical model with **physical semantics** lets you reason without profiling — "what if I double context length?" is now a formula, not an experiment. It also makes architecture research cheaper: when evaluating a new attention variant, you can predict serving cost before implementing the kernel. The 3.1–3.5× accuracy win over prior models matters precisely because planning is sensitive to small errors (a 10% misestimate compounds across a fleet).

## Gotchas & tricks

- Fits assume a specific serving stack (vLLM/SGLang-shape with continuous batching). Switching to a different scheduler (e.g. prefill/decode disaggregation) changes $\delta$ and the coupling between prefill and decode.
- The quadratic prefill term dominates at long prompts but is attenuated by FlashAttention's memory pattern; the paper's fit absorbs that into $\beta_p$ — don't try to interpret it as pure $P^2$ FLOPs.
- MoE models break the per-token compute assumption — $\alpha_d$ becomes a function of the active-expert count. Extend by splitting $\alpha_d$ into a routing-cost term and an expert-FLOPs term.
- Quantization changes $\gamma$ (KV bytes/tok) and $\alpha_d$ (matmul FLOPs/tok) by known factors — the law composes with quantization models cleanly.

## Sources

- Paper: *The SIMPLE Law for Energy Cost of LLM Inference* — Pizzini Cavagna et al., 2026 — https://arxiv.org/abs/2611.03330
- Related: *Roofline models for LLM inference*, prior per-model cost regressions (POLCA, Splitwise).
