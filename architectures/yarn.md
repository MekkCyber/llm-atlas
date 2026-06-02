# YaRN — Yet another RoPE extensioN

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A context-window extension technique for RoPE-based transformers. Given a model trained at sequence length $L_\text{train}$, YaRN re-scales the RoPE frequencies so the model can attend to positions $> L_\text{train}$ without retraining from scratch — but does so *non-uniformly* across the frequency spectrum, leaving high-frequency components alone and aggressively interpolating only the low-frequency ones. Two-stage process: a frequency-band-aware interpolation (NTK-by-parts), plus an attention-temperature correction that compensates for the changed expected attention entropy at longer sequences. A short fine-tune (~1K steps) on long-context data locks in the extension.

**Prereqs:** [rope](../fundamentals/rope.md), [_positional-encoding](../fundamentals/_positional-encoding.md)
**Related:** [dca](../fundamentals/dca.md) · [sliding-window-attention](sliding-window-attention.md)

---

## What it is

RoPE applies a fixed rotation in each query/key 2D subspace, with the rotation rate $\theta_i$ chosen geometrically per dimension. At sequence length $L_\text{train}$, the model has only ever seen position differences $\Delta = 0, 1, \ldots, L_\text{train} - 1$, so it never observed how attention behaves at $\Delta > L_\text{train}$.

The naive extension — let the model evaluate at $\Delta > L_\text{train}$ — fails badly: the high-frequency rotations wrap into nonsense angles the model never trained on, and attention scores collapse.

Earlier fixes (Position Interpolation, NTK-aware scaling) re-scale $\Delta$ down so the longer evaluation sequence fits inside the trained $\Delta$ range. YaRN is the refinement: it applies *different* scaling rules to *different* RoPE frequency bands, and adds an attention-temperature correction.

---

## How it works

### Frequency-band classification

For each RoPE dimension $i$, compute the implied wavelength at training:

$$
\lambda_i = \frac{2\pi}{\theta_i}
$$

Dimensions partition by how their wavelength compares to $L_\text{train}$:

- **High-frequency** ($\lambda_i \ll L_\text{train}$): the model saw many full rotations during training. Extrapolation is safe — these dimensions encode local position cleanly. **Leave alone.**
- **Low-frequency** ($\lambda_i \gtrsim L_\text{train}$): the model never completed a rotation during training; these dimensions encode "approximate absolute position." **Interpolate** (compress the position scale).
- **Middle-frequency**: a smooth ramp between the two.

The ramp function $\gamma(i)$ controls how much each band is interpolated. Setting it as a smooth step around the bands gives the "NTK-by-parts" scaling.

### Attention temperature correction

Even with the right interpolation, longer sequences see more queries × more keys, which shifts the expected attention-softmax entropy and miscalibrates the model's learned attention scale. YaRN multiplies softmax inputs by a small correction factor $1 / t$ where $t = 0.1 \cdot \ln(s) + 1$ for scale factor $s = L_\text{new} / L_\text{train}$. This is empirically derived and is the difference between YaRN and pure NTK-by-parts.

### Short fine-tune

After applying YaRN, a brief fine-tune (often ~1K–2K steps on long-context data) lets the model adapt the residual mismatch. This is much cheaper than retraining from scratch and lets the model lock in the extended positional behavior.

### Layer-selective YaRN (modern usage)

In mixed-attention models (SWA-on-most-layers + full-attention on a few), YaRN is often applied only to the **full-attention layers**, since the SWA layers' effective range is already bounded by the window — there's no long-range positional information to interpolate. See [Mellum 2 case study](../case-studies/mellum-2.md).

---

## Why it matters

- **Cheap context extension.** Going from 4K → 32K → 128K via YaRN+fine-tune costs orders of magnitude less than retraining with the long context from scratch.
- **Better than naive interpolation.** The frequency-band-aware rule preserves local-position fidelity (which uniform interpolation flattens) while extending the model's long-range reach.
- **Default open-recipe context extension.** Used by Llama 3, DeepSeek-V3 (32K → 128K), Qwen2.5, Mellum 2, and most modern open releases that ship a long-context variant.

---

## Gotchas & tricks

- **Don't forget the temperature correction.** Pure frequency-band scaling without the $1/t$ correction underperforms full YaRN on long-context evals.
- **Fine-tune is not optional.** YaRN-without-fine-tune gives a quick prototype but loses noticeable quality vs YaRN-with-fine-tune at the extended context.
- **Scale factor compounds in stages.** Going 4K → 128K in one step often underperforms staged extensions (e.g., 4K → 32K → 128K). DeepSeek-V3 uses two YaRN stages explicitly.
- **Layer-selective for mixed-SWA models.** Applying YaRN to layers that don't see far enough to use it is wasted effort and can introduce mild noise. Skip the SWA layers.
- **Inference KV-cache size still grows.** YaRN only changes the positional encoding; the KV cache at 128K is still 32× larger than at 4K. Combine with [SWA](sliding-window-attention.md) or KV-compression to actually run cheaply.

---

## Sources

- Paper: *YaRN: Efficient Context Window Extension of Large Language Models* — Peng, Quesnelle, Fan, Shippole, 2023, [arXiv 2309.00071](https://arxiv.org/abs/2309.00071) — introduces the frequency-band scaling and attention-temperature correction.
- Paper: *Position Interpolation* — Chen, Wong, Chen, Tian, 2023, [arXiv 2306.15595](https://arxiv.org/abs/2306.15595) — predecessor that YaRN refines.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — two-stage YaRN extension to 128K.
- Paper: *Mellum 2* — JetBrains, 2026 — layer-selective YaRN on full-attention layers only in a mixed-SWA model.
