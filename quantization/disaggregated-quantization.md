# Disaggregated Quantization (DQ)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Prefill and decode want opposite quantization regimes — prefill benefits from low-precision arithmetic (compute-bound), decode benefits from compact weights (memory-bound). DQ trains **two** quantized copies of the same model, one specialized per phase, and swaps between them inside a disaggregated serving stack. Weight formats are chosen per phase; activation quantization is dropped on decode.

**Prereqs:** [_number-formats](./_number-formats.md), [prefill-decode-disaggregation](../inference/prefill-decode-disaggregation.md)
**Related:** [fp8](./fp8.md)

---

## What it is

Standard PTQ quantizes a model once and uses the same weights for prefill and decode. That is a bad tradeoff: on decode, tensor cores are wildly underused, and any accuracy lost to activation quantization is wasted (because compute isn't the bottleneck); on prefill, aggressive 2–3-bit weight compression hurts arithmetic throughput.

Disaggregated Quantization keeps a **separate, phase-specialized checkpoint** for each phase:

- **Decode checkpoint** — weight-only quantization, aggressive (2–3-bit). Activations stay in high precision because they cost nothing to store and everything to compute correctly.
- **Prefill checkpoint** — compute-native format (NVFP4, FP8, INT8) with matching activation quantization; trained (or PTQ'd) *independently* to match compute-heavy kernels.

Both checkpoints reference the same logical model; a router selects the correct one for each phase.

## How it works

1. **Start from a decode checkpoint** (an existing weight-only quantized model, e.g. a released GGUF).
2. **Train a compute-native prefill copy** whose logits match the decode checkpoint on a calibration set — a form of teacher-student distillation, but between two quantized versions of the same base.
3. **Deploy on a disaggregated serving stack** (vLLM, llama.cpp): prefill workers load the prefill checkpoint, decode workers load the decode checkpoint. KV cache produced by prefill flows to decode as usual.
4. **Offloaded Disaggregated Prefill (ODP)** — when only one device is available, stream the prefill weights from SSD during prompt processing, amortizing load cost over prompt length.

Format specialization is orthogonal to serving disaggregation: DQ builds on top of it but adds the weight-format axis.

## Why it matters

- **Removes the "single quantization" straitjacket.** Every previous PTQ recipe picked one operating point; DQ picks two, gets both.
- **Massive low-bit gains.** On Qwen3-27B GGUF: training an NVFP4 prefiller improves 1-bit decode accuracy by **+32.5 pts on MMLU-Pro and +35.3 on MMMU-Pro** with no change to the decode checkpoint itself.
- **Cost-neutral TTFT wins.** ODP delivers 1.78× TTFT speedup over the weight-only baseline at 8K prompt length in llama.cpp on a single device.
- **Validated at scale.** Post-training format disaggregation shown up to 2.8T parameters.

## Gotchas & tricks

- **Two checkpoints ≈ 2× disk.** Not free, but SSD is cheap; and per-device you only need one at a time (ODP streams).
- **Activation drop on decode is the biggest single knob.** If you take nothing else from DQ, remove activation quantization from the decode path.
- **Prefill weights must match the decode logits.** They aren't independently fine-tuned; they distill against the decode checkpoint so their outputs are consistent for the same prompt.
- **KV cache format is a separate decision.** DQ doesn't change how KV is stored or transferred; combine with KV quantization schemes (e.g. FP8 KV) independently.
- **Applies to any weight-only quantization family.** GPTQ, AWQ, MXFP4 decode all pair fine with an NVFP4/FP8 prefill.

## Sources

- Paper: *Disaggregated Quantization: Specializing LLM Prefill and Decode* — Panferov, Kleinegger, Priyadarshi, Blankevoort, Alistarh — ISTA + Qualcomm AI Research, 2026 — https://arxiv.org/abs/2609.26333
- Related: *DistServe* / *Splitwise* — see [prefill-decode-disaggregation](../inference/prefill-decode-disaggregation.md) — the serving-side disaggregation DQ builds on.
- Related: NVIDIA NVFP4 microscaling spec (Blackwell).
