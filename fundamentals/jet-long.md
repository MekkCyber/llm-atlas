# Jet-Long — Dynamic Bifocal RoPE
*Depth — a tuning-free zero-shot long-context extension for RoPE models.*

**TL;DR:** Existing zero-shot RoPE extension methods (PI, NTK-aware, YaRN, ABF) commit to one rescaling factor for the whole model — aggressive factors degrade short-context fidelity, conservative ones fail at long contexts. Jet-Long removes the commitment: attention is split into a **local RoPE-faithful window** (base frequencies unchanged, matches the pretrained model exactly at short inputs) and a **long-range window** whose rescaling factor *adapts dynamically to the current sequence length*. "Bifocal" = two focal lengths on RoPE frequencies applied simultaneously. Fully tuning-free.

**Prereqs:** [rope.md](rope.md), [_positional-encoding.md](_positional-encoding.md), [attention.md](attention.md)
**Related:** [dca.md](dca.md) · [_rope-extension.md](_rope-extension.md)

---

## What it is

A drop-in modification of the attention block that computes attention twice per head, once under two different RoPE parameterizations, and combines the results. The local branch uses the pretrained base (typically `base = 10000` or `500000`); the long-range branch scales the frequency spectrum by a factor that is a function of the *observed* sequence length at inference. Because the local branch is exactly the pretrained configuration, base-model behavior is recovered whenever the input is short — a property none of the fixed-factor methods (PI / YaRN / ABF) can guarantee.

## How it works

Given input sequence length $S$ at inference, Jet-Long constructs two rotary tables:

1. **Local table:** identical to the pretrained RoPE — $\theta_i = \mathrm{base}^{-2i/d}$.
2. **Long-range table:** rescaled with a length-dependent factor $s(S) = f(S / L_\text{train})$ so that at $S = L_\text{train}$ the factor is 1 (no rescaling) and grows monotonically with $S$. The exact shape is monotonic and continuous — it degenerates to the pretrained parameterization at short inputs and to a YaRN-like aggressive rescaling at long ones.

Attention then runs as two RoPE variants of the same queries and keys, fused per-head or per-position via a gating / mixture rule. The critical property: both branches see the same $Q$ and $K$ *content* — the only difference is which position-dependent rotation is applied. No weights are added; no fine-tuning is required.

## Why it matters

- **No short-context regression.** YaRN, PI, and ABF all subtly distort attention on inputs shorter than the training length because they alter the frequency spectrum globally. Jet-Long's local branch guarantees exact recovery of the pretrained model on short inputs.
- **Adaptive to input length.** Fixed-factor methods force a choice between "good at 32k, bad at 128k" and "good at 128k, degraded at 8k." A length-dependent factor removes the tradeoff.
- **Tuning-free deployment.** Every open-weights checkpoint can adopt Jet-Long without fine-tuning — the dominant deployment path for long-context extension.

## Gotchas & tricks

- The "bifocal" fusion doubles attention compute per token; overhead is small because RoPE is cheap, but the constant-factor cost is real.
- The length-adaptive factor $s(S)$ must be monotonic *and* continuous through $S = L_\text{train}$, otherwise attention scores jump when a new token pushes $S$ over the boundary.
- Composes with FlashAttention and MLA (rotary layout is unchanged in each branch).
- Not a substitute for supervised long-context fine-tuning when maximum quality at extreme lengths matters — Jet-Long is the *zero-shot* frontier, not the peak.

## Sources

- Paper: *Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE* — Han Cai et al., NVIDIA, 2026 — [arXiv:2607.07740](https://arxiv.org/abs/2607.07740).
- Prior: *YaRN* — Peng et al., 2023 — the fixed-factor method Jet-Long generalizes.
- Prior: *Position Interpolation* — Chen et al., 2023 — the earliest RoPE post-hoc rescaling.
