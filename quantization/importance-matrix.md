# Importance Matrix (imatrix) Calibration
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A calibration mechanism in the llama.cpp / GGUF quantization stack. The *importance matrix* stores per-weight importance scores computed from activations on a calibration corpus; the quantizer then protects high-importance weights with more bits (or shrinks their quantization error target) while compressing low-importance weights more aggressively. Without an imatrix, low-bit GGUF quantization can produce **non-functional** artifacts — the 2026 "Broken on Arrival" audit found 2.3% of the official Ollama library's code-capable low-bit tags scored ≤6/164 on HumanEval+, and the fix on five audited failures was adding imatrix calibration during quantization.

**Prereqs:** [_number-formats](_number-formats.md)
**Related:** [fp8](fp8.md)

---

## What it is

Classical post-training quantization minimizes per-tensor quantization error uniformly. That's fine at 8-bit on 70B models, but at **low bit widths on small models** (≤4B at 2–4 bit), uniform error minimization has nothing to spend on which weights actually drive outputs. The imatrix is the "importance signal" that decides where to spend the bits.

In llama.cpp's GGUF stack, generating an imatrix is a specific preprocessing step: run the full-precision model over a calibration corpus, log per-weight activation-driven importance, write the imatrix file. The quantizer then consumes it and reweights the per-weight loss during quantization.

## How it works

- **Pass 1: calibration.** Run the fp16/bf16 model over a calibration corpus (typically a few hundred sequences, mixed English + code). Capture statistics on how each weight's output contributes to the loss.
- **Pass 2: quantization with weighted loss.** The per-weight quantization error is weighted by its imatrix entry. High-importance weights get smaller allowable error (closer to original); low-importance weights carry more of the quantization error budget.
- **Independence from the weights file.** The imatrix is a side file; the same base checkpoint can be quantized with different imatrices derived from different calibration sets.

## Why it matters

The "Broken on Arrival" audit (2026) is the strongest practical argument:

- 305 code-capable GGUF tags in the official Ollama library were screened; 7 (2.3%) solved ≤6/164 HumanEval+ tasks.
- All failures were low-bit quantizations of ≤4B-parameter models.
- Tensor-for-tensor reproducible from the original weights → **not corruption, just undercalibrated**.
- Adding an imatrix during requantization took three Qwen2.5-Coder-3B builds from **0 to 76–113 solved tasks** on HumanEval+.
- The same uncalibrated recipe left Qwen2.5-Coder-1.5B functional — i.e., failure is model-specific, not recipe-wide.

Takeaway for registry publishers: **screen low-bit quants on a functional eval before publishing, and calibrate (imatrix) or withhold the failures.**

Takeaway for consumers: on low-bit tags of small models, prefer builds with a published imatrix.

## Gotchas & tricks

- **Calibration corpus matters.** An English-only imatrix for a coding model will underweight code paths. Mix the calibration corpus to the model's intended use.
- **Low-bit is where it matters most.** At 8-bit, imatrix helps marginally. At 4-bit on small models, it can be the difference between functional and dead.
- **Exact reproducibility ≠ functional.** The audit showed suspect files matched a clean rebuild tensor-for-tensor. Deterministic reproduction does not catch calibration failures — a functional screen (e.g., HumanEval+ for code models) does.
- **Model-size × bit-width interaction.** A 1.5B model may survive uncalibrated 4-bit; the same recipe on a 3B model collapses. The paper's inference-time functional test is the only reliable predictor.
- **Not specific to GGUF.** The *concept* (importance-weighted quantization) applies to GPTQ, AWQ, and other PTQ methods, each with their own importance-estimation step. The 2026 paper focuses on the GGUF/llama.cpp instance because that's what public registries host.

## Sources

- Paper: *Broken on Arrival: Non-Functional Quantized LLM Artifacts in Public Model Registries and How to Catch Them* — A. Patodiya, 2026 — arXiv:2609.05881 — audit motivating imatrix hygiene.
- Code: https://github.com/aditi-p31/quantcheck — the audit's screening tool.
- Project: llama.cpp — `llama-imatrix` tool, upstream implementation of imatrix calibration.
