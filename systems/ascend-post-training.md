# Ascend Post-Training for Trillion-Parameter MoEs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Full-parameter post-training of trillion-parameter MoEs on non-NVIDIA silicon is a systems problem: memory pressure, non-overlapped all-to-all communication, and unfused kernels. SLAI T-Rex is an end-to-end stack for the Huawei Ascend NPU SuperPOD applied to DeepSeek-V4 CPT+SFT, combining tiered activation offload, all-to-all + expert-compute overlap, and NPU-native fused kernels. Reaches **34.22% MFU** — a 2.93× improvement over their baseline. First public reference point for post-training DeepSeek-V4.

**Prereqs:** [dualpipe.md](./dualpipe.md), [../architectures/_moe.md](../architectures/_moe.md), [../pre-training/fp8-training.md](../pre-training/fp8-training.md)
**Related:** [../case-studies/deepseek-v3.md](../case-studies/deepseek-v3.md), [../architectures/deepseek-moe.md](../architectures/deepseek-moe.md), [ray.md](./ray.md)

---

## What it is

DeepSeek-V3-class MoE post-training was written for NVIDIA H800 clusters using DualPipe and CUDA-native kernels. Porting it to Huawei's Ascend NPU SuperPOD — a serious frontier alternative outside the CUDA moat — hits three concrete problems:

1. **Memory pressure.** Full-parameter (not LoRA) post-training keeps optimizer state + activations for a trillion-parameter model in NPU HBM. NPU memory hierarchies are different from H800; naive porting OOMs.
2. **Non-overlapped all-to-all.** Expert parallelism requires all-to-all comm per layer. On Ascend without DualPipe-style scheduling, the collective becomes the wall-clock bottleneck.
3. **Unfused kernels.** Native Ascend kernels for MoE routing, expert compute, and FP8 accumulation didn't exist at DeepSeek-scale performance until SLAI built them.

## How it works

**Memory layer.** Tiered activation offload (HBM → host DRAM) plus selective activation recompute keyed to per-layer memory pressure. Full-parameter optimizer state stays in HBM; activations are the offloaded axis.

**Communication layer.** Fused all-to-all + expert-compute pipeline that hides expert-parallel comm behind local matmul time. The construct is Ascend's analog of DualPipe — same motivation (overlap comm with compute across the MoE forward), different scheduling primitives for the NPU interconnect topology.

**Kernel layer.** NPU-native fused kernels for the MoE router (top-k gate + token dispatch), the expert compute (grouped GEMM), and the FP8 accumulation path used across the model.

**Application.** Applied to DeepSeek-V4 CPT (continual pretraining) and SFT for Operations Research tasks. The specialized OR models reach 71.81% zero-shot Pass@1, beating GPT-5.4-Mini by 3.98 pp — evidence the systems work delivers actual training throughput, not just synthetic MFU numbers.

## Why it matters

- **Frontier MoE training outside CUDA is real.** 34.22% MFU on trillion-param full-parameter post-training is competitive with published NVIDIA numbers for the DeepSeek-V3 recipe. This changes the "you must use CUDA" default for open-source labs.
- **First public DeepSeek-V4 post-training reference.** The V4 base model is now a research target with a documented Ascend recipe.
- **Template for hardware ports.** The three-layer (memory / comm / kernels) breakdown is generalizable to other non-NVIDIA backends (Trainium, TPU, Cerebras).

## Gotchas & tricks

- 34.22% MFU is with their fully-tuned stack; naive Ascend porting produced 11.7% MFU — most of the gap is in kernel tuning, not the algorithm.
- Tiered offload works because CPT/SFT tolerates the extra host-DRAM traffic; RL post-training with tighter memory budgets may need a different scheme.
- Not open-source (at time of writing); the paper is a systems report, not a released framework.
- The OR benchmark is domain-specific; do not extrapolate the "beats GPT-5.4-Mini" claim to general reasoning.

## Sources

- Paper: *SLAI T-Rex: Full-Parameter Post-training of the DeepSeek-V4 Family on Ascend SuperPOD* — Li et al. (65+ authors), 2026 — [arXiv:2607.20145](https://arxiv.org/abs/2607.20145)
- Prior art: *DeepSeek-V3 Technical Report* — the H800 reference stack SLAI T-Rex ports and extends. See [../case-studies/deepseek-v3.md](../case-studies/deepseek-v3.md).
