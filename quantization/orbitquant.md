# OrbitQuant
*Depth — a data-agnostic weight-activation quantizer for diffusion transformers.*

**TL;DR:** Post-training quantization for DiTs needs a calibration set — a batch of real prompts, timesteps, and CFG branches to estimate activation ranges. Prior methods (GPTQ, AWQ, SmoothQuant) re-fit this per checkpoint, per modality, per fine-tune. OrbitQuant sidesteps calibration by applying an orthogonal rotation to weights and activations before quantization. In the rotated basis, the range distribution is *predictable independent of data*, so the quantizer can round without ever having seen a real activation. One quantizer works across image and video DiTs, no per-model data.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** [../pre-training/fp8-training.md](../pre-training/fp8-training.md)

---

## What it is

Diffusion transformers (DiTs) drive state-of-the-art image and video generation but are expensive: sampling requires 20–50 forward passes per image, and models keep growing (Sora-scale hits tens of billions of parameters). Post-training quantization (PTQ) is the standard reply, but DiT activations shift systematically across three axes:

- **Timesteps.** Early and late denoising steps have very different activation scales.
- **Prompts.** Different text conditions produce different activation distributions.
- **CFG branches.** Classifier-free-guidance runs a conditional and unconditional pass with different scales.

Every published PTQ method calibrates by running the model over a representative batch, measuring per-channel min/max, and setting scales. Change the checkpoint or the modality and you re-run calibration. OrbitQuant removes this dependency: quantize in a rotated basis where range estimation is unnecessary.

## How it works

**The rotation trick.** For a linear layer $y = Wx$, insert an orthogonal matrix $R$: $y = W R^\top R x = (W R^\top)(R x)$. The linear layer is unchanged, but $\tilde W = W R^\top$ and $\tilde x = R x$ have different statistics.

Choose $R$ such that outliers in $x$ — those handful of channels that carry huge magnitudes and dominate quantization error — are *spread* across all rotated channels. The Hadamard transform is a common choice; OrbitQuant uses a similar structured rotation. In the rotated basis, the distribution of activation magnitudes concentrates around a predictable "orbit" — mean-centered, roughly Gaussian, with tails that don't depend on the specific data.

**Quantize in the rotated basis.** With outliers absorbed, INT/FP4 rounding produces low error without needing real activations to fit scales. The quantizer uses a *normalized* range — derived from theoretical properties of the rotation, not from calibration data.

**Same recipe across models.** Because the rotation depends only on the layer structure (not on the data), applying it to a new DiT checkpoint or a different modality (image DiT vs. video DiT) does not require refitting. That is the "data-agnostic" claim.

## Why it matters

- **Calibration data was a real production cost.** Every new DiT fine-tune (aspect ratio, style, LoRA merge) forced a fresh calibration run. Skipping it removes a slow, brittle step from serving pipelines.
- **Cross-modal transfer for free.** The same quantizer moves from image to video DiTs without any refit. Practical for teams that ship both.
- **Sits in a growing family of rotation-based methods.** QuaRot, SpinQuant, and Hadamard-QAT all use rotations for outlier absorption; OrbitQuant is the DiT-specific member. The design pattern — rotate, then round — is becoming the dominant recipe for aggressive weight+activation PTQ.
- **Compatible with runtime kernels.** The rotation can be folded into adjacent weight matrices at load time, so there is no per-inference rotation cost.

## Gotchas & tricks

- **Rotation quality depends on layer statistics.** If a specific layer has extreme structured outliers (e.g. a single dominant channel in AdaLN modulation), the theoretical range assumption weakens and per-layer fallback calibration may still help.
- **Folded-in rotations vs. runtime rotations.** Folding $R$ into the previous matrix eliminates runtime cost but requires care around residual connections and layer norms — the rotation must be inverted before the residual add. Prior work (QuaRot) documents the invariance conditions.
- **Weight-only vs. weight+activation.** Weight-only quantization is easier and often gets 80% of the benefit; OrbitQuant targets full weight+activation because activations are the actual bottleneck on modern hardware (H100 tensor cores) at DiT scale.
- **Diffusion-specific timestep dependence.** Even rotated, activation ranges can drift across the denoising trajectory. OrbitQuant's data-agnostic claim rests on the rotated distribution being *stable enough* across timesteps that a single scale works — worth measuring on your target checkpoint.

## Sources

- Paper: *OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers* — Lee et al., Cantina Labs / USC / UIUC / Yale, 2026 — [arXiv:2607.02461](https://arxiv.org/abs/2607.02461)
- Related: *QuaRot* (Ashkboos et al., 2024), *SpinQuant* (Liu et al., 2024) — rotation-based PTQ for LLMs.
