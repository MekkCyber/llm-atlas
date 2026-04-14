# Quantization

*Reducing numerical precision to trade model quality for memory, throughput, and cost — formats, calibration strategies, and the system implications.*

---

## What This Is

Quantization is how a model that needed an H100 to train fits on a 24GB consumer GPU at inference, or runs 4× faster on the same hardware. This folder covers the numerical formats (FP8, INT8, INT4, MXFP4, NF4), the algorithms that pick scales and rounding (GPTQ, AWQ, SmoothQuant), and the tradeoffs against quality.

---

## What Belongs Here

- **Number formats** — FP16, BF16, FP8, INT8, INT4, NF4, MXFP4 and friends.
- **Post-training quantization (PTQ)** — GPTQ, AWQ, SmoothQuant, calibration.
- **Quantization-aware training (QAT)** — training with quantized weights/activations.
- **Mixed-precision strategies** — which layers stay high precision, which don't.
- **Extreme quantization** — BitNet, ternary, sub-bit.
- **Quantized fine-tuning** — QLoRA and friends (also see [post-training/fine-tuning/](../post-training/fine-tuning/)).
- **Hardware considerations** — what each GPU/accelerator natively supports.

## Reading Order

1. Number formats overview (FP16/BF16/FP8/INT8/INT4)
2. PTQ basics & calibration
3. GPTQ
4. AWQ & SmoothQuant
5. QAT
6. Extreme quantization (BitNet)

---

## Related

- [inference/](../inference/) — quantization is mostly an inference optimization.
- [pre-training/](../pre-training/) — mixed precision in training (FP8, BF16).
- [post-training/fine-tuning/](../post-training/fine-tuning/) — QLoRA sits here.
