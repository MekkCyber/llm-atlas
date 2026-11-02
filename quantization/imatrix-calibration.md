# Importance-Matrix (imatrix) Calibration

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For low-bit (≤4-bit) GGUF quantization, the raw min/max-based block scaling catastrophically breaks small models — a non-trivial fraction of publicly-indexed quantized models in Ollama and Hugging Face are *bit-perfect vs their published recipe* yet produce incoherent text. The fix is cheap: compute an "importance matrix" (`imatrix`) from a small calibration dataset and use it to weight the per-block quantization error. Adding `imatrix` restores function; a 15-task functional screen catches the regressions in ~30 seconds.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** [fp8](fp8.md)

---

## What it is

GGUF's low-bit formats (Q2–Q4, including Q4_K_M, Q3_K_S, Q2_K) quantize weights block-wise with a single scale per block. "Blind" quantization computes that scale from the raw min/max of each block. For large models the resulting quantization error averages out; for small (≤4B) models at ≤4-bit, a few blocks with heavy-tailed weights can dominate output quality, producing a model that *compiles* and *runs* but outputs noise.

The fix, borrowed from GPTQ-style calibration, is to weight the quantization error of each weight by its *importance* on a calibration set:

```
imatrix[i, j] = mean over calibration tokens of ( activation_input[j]^2 )
```

For a layer `W @ x`, weighting the reconstruction error by `imatrix` yields a per-block scale that preserves outputs on the calibration data.

## How it works

### Producing the imatrix

1. Pick a small calibration corpus (a few MB of mixed text and code).
2. Run the un-quantized model on this corpus and record the squared input activations per weight column at each layer.
3. Save the per-column importances as the `imatrix`.

### Quantizing with imatrix

4. For each block of weights, choose the scale/offset that minimizes **imatrix-weighted** quantization error rather than raw min/max error.
5. The quant kernel is unchanged — only the chosen scales differ.

### Catching failures

The paper proposes a **15-task functional screen** (≤30 seconds per model): a tiny suite of basic generation prompts whose expected outputs are stereotyped (format compliance, repetition, simple arithmetic). A model that fails any of these is non-functional; publish the gate alongside the model card.

## Why it matters

- Model registries ship millions of downloads of quantized files whose quality nobody re-checks. The paper measured a ~2–3% failure rate on official Ollama tags and found similar in community HF files, all concentrated at low bit × small model.
- The fix is a one-line addition to the quantization recipe. There is no technical reason broken low-bit builds should still ship.
- Argues for a release-time gate (functional screen) as a cheap release-process control, not as a benchmark replacement.

## Gotchas & tricks

- The imatrix is calibration-set-dependent. A calibration corpus heavy on English will underweight code tokens; mix domains if the deployed model will be used across them.
- `imatrix` helps most for Q2–Q4. Q5/Q6/Q8 rarely need it because the scale resolution is already high enough.
- Importance values vary across layers by orders of magnitude; clamp or normalize per layer to avoid single-layer dominance.
- Adding `imatrix` does not fix a *model* that is intrinsically fragile at low bit — some architectures (shallow MoEs, narrow MLPs) remain brittle below Q4 regardless. Sanity-check with the functional screen.
- The failure modes studied are *reproducible from the published recipe* — i.e. the file is not corrupted. "Re-download" doesn't help; the recipe itself is wrong.

## Sources

- Paper: *Broken on Arrival: Non-Functional Quantized LLM Artifacts in Public Model Registries and How to Catch Them* — Patodiya, 2026 — [arXiv:2609.05881](https://arxiv.org/abs/2609.05881).
- Related: *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — Frantar et al., 2023 — the activation-weighted quantization-error idea.
- Reference: `llama.cpp` `imatrix` tool — [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — standard pipeline.
