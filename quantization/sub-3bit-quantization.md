# Sub-3-bit Weight Quantization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Compress LLM/VLM weights to below 3 bits per parameter to fit on mobile-class hardware. **Llama-Mobile** (Ribar, Bhoot, Orr, 2026) uses a **2.7-bit** weight format optimized for Arm CPU execution paths, paired with a **self-generated calibration pipeline** — the model synthesizes its own calibration data, so no access to the original training corpus is required. Compresses Llama 3.2 11B Vision Instruct to 3.7 GB (8-bit activations) with preserved VQA performance.

**Prereqs:** [_number-formats.md](./_number-formats.md), [fp8.md](./fp8.md)
**Related:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md) · [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Below-3-bit weight quantization is the frontier at which quantization stops being a pure memory-halving stopgap and starts changing what fits on-device. The regime is distinct from 4-bit (INT4, MXFP4, NF4) because:

- Individual channels can't tolerate uniform quantization — outlier-aware or non-uniform schemes are mandatory.
- The choice of *exact* fractional bit-width is dictated by hardware instruction paths, not by rounding theory.
- Calibration data quality dominates accuracy: too little, and outliers are missed; wrong distribution, and preserved accuracy is illusory.

## How it works

Two ingredients from Llama-Mobile:

**2.7-bit weight format.** Fractional-bit formats are constructed by packing groups of weights so that the *effective* bits-per-weight matches the hardware's throughput sweet spot. On Arm CPUs, particular SIMD-lane widths and shuffle patterns favor packings that are neither 2 nor 3 bits per element. 2.7-bit lands where the Arm instruction sequence for dequantize-then-matmul is shortest. Concrete: values are stored in shared-scale blocks; per-block metadata + per-element indices average to 2.7 bits/weight.

**Self-generated calibration.** Rather than reusing the training corpus (which for a released VLM is usually unavailable), the pipeline prompts the *quantized-target model itself* to generate calibration prompts and completions across relevant task distributions. The generated data is used to (i) select per-channel scales, (ii) detect outlier channels for higher-precision fallback, and (iii) fit block-wise scales to observed activation ranges.

Activations remain 8-bit (per-token dynamic quantization).

## Why it matters

- **Mobile deployment ceiling.** An 11B-param VLM at 3.7 GB with 8-bit activations fits on flagship mobile RAM budgets with room for the runtime. INT4 at ~5.5 GB does not.
- **Training-data-independent pipeline.** For third-party quantizers (users, downstream deployers), the training set is rarely available. Self-generated calibration removes that barrier.
- **Format economics.** Establishes that hardware-optimal precision is not always a whole number of bits — a claim that will generalize to other Arm/x86/mobile-NPU vendors.

## Gotchas & tricks

- **Fractional-bit formats break naive packing assumptions.** Layer weights don't align to byte boundaries, so kernels must handle sub-byte layouts throughout the matmul loop.
- **Self-generated calibration ≠ zero-shot.** The pipeline still needs a target-task descriptor (e.g. "visual QA"); it just doesn't need the original training data. Bias in the target descriptor becomes bias in the calibration set.
- **Not all Arm SoCs are equal.** The 2.7-bit sweet spot is instruction-path-specific — porting to different vector widths (Cortex-A vs Neoverse vs custom NPUs) may require a different fractional bit width.
- **Activation quantization is separate.** Weight-only compression to 2.7-bit leaves activations at 8-bit; you don't get another 3× on runtime memory from activation quant without paying accuracy.
- **Not a replacement for training-time low-precision.** Post-training weight-only quantization can't recover from architectural sensitivity — models trained natively in low precision (FP8, MXFP4) can go lower still.

## Sources

- Paper: *Llama-Mobile: Efficient 2.7-Bit Quantization of VLMs* — Ribar, Bhoot, Orr (Graphcore), 2026.
- Paper: *GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers* — Frantar et al., 2022 — INT4 baseline.
- Paper: *QLoRA: Efficient Finetuning of Quantized LLMs* — Dettmers et al., 2023 — NF4 non-uniform 4-bit.
- Paper: *AWQ: Activation-aware Weight Quantization* — Lin et al., 2023 — outlier-aware weight quant.
