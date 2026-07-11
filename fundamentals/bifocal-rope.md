# Bifocal RoPE
*Depth — dynamic dual-window RoPE for zero-shot long-context extension.*

**TL;DR:** A tuning-free RoPE modification that pairs a **local** window at the model's native frequencies with a **long-range** window whose rescaling factor is chosen dynamically from the current sequence length. At short inputs the local window recovers the base model exactly (unlike YaRN or NTK-aware scaling, which cost short-context accuracy up front). At long inputs the long-range window extrapolates cleanly. Introduced by Jet-Long (NVIDIA, 2026).

**Prereqs:** [rope.md](./rope.md), [attention.md](./attention.md)
**Related:** [_positional-encoding.md](./_positional-encoding.md), [dca.md](./dca.md)

---

## What it is

Zero-shot context extension for a RoPE-pretrained LLM without any fine-tuning or distillation. The standard tools (Position Interpolation, NTK-aware scaling, YaRN, ABF) each pick **one static rescaling factor** at deploy time. Aggressive factors sacrifice short-context fidelity; conservative factors break down at very long context. Bifocal RoPE removes the tradeoff by running two RoPE views inside every attention head and dynamically re-choosing the long-range factor at each input.

Sits in the same rotary-extension family as YaRN and NTK-aware but is the first to make the rescaling factor a **function of the current sequence length** rather than a global config.

## How it works

Each attention head runs two rotary transforms in parallel:

1. **Local RoPE-faithful window.** Applies the unmodified rotation $R(m)$ using the pretraining base frequency. When the input length $L$ ≤ pretraining window $L_0$, only this window is active — so the base model is recovered exactly, no quality lost.
2. **Long-range window with dynamic rescale.** Applies a rescaled rotation $R(m \cdot s(L))$ where $s(L) = f(L / L_0)$ is a monotonically-shrinking function of the current sequence length. For $L \gg L_0$, $s$ shrinks so the maximum angle stays inside the trained range (the same trick as PI/YaRN), but $s$ is picked **per input**, not per deployment.

The two outputs are combined into a single attention score (paper details the mixing). Because both windows share the underlying rotation machinery, there's no extra parameters, no fine-tuning, and no incompatibility with FlashAttention or GQA / MLA.

Compare to sibling techniques:

- **PI / YaRN / NTK-aware:** one static rescale; must trade short vs long.
- **DCA (Dual Chunk Attention):** splits attention *by chunk boundaries*; bifocal RoPE splits *by frequency window*.
- **ABF (base-frequency scaling):** pick base = 1M and hope; bifocal picks it dynamically.

## Why it matters

Modern LLM deployment is bimodal: the same open-weight checkpoint serves short chat traffic and long RAG / repo-level / agentic-trace traffic on the same GPU. Existing extension methods force a single global choice that hurts one side of the traffic mix. Bifocal RoPE lets a serving stack keep the base model exact for short inputs while extending cleanly to sequences an order of magnitude past the pretraining window — zero-shot, no separate long-context checkpoint required.

## Gotchas & tricks

- **Local window recovery is the headline property.** If short-context accuracy drops after applying bifocal RoPE, the local-window fusion is wired wrong — the base model must recover exactly at $L \le L_0$.
- **Rescale schedule matters.** The paper picks $s(L)$ empirically; wrong shape reintroduces the exact tradeoff bifocal was designed to avoid.
- **Composes with attention kernels.** Because both views are rotations, FlashAttention / GQA / MLA all keep working — but the mixing step must run in the attention kernel, not as a post-hoc combination in float32, to preserve throughput.
- **Zero-shot only.** No fine-tuning is claimed; if fine-tuning is available, YaRN with a short adapter can still edge it out at the extreme tail.

## Sources

- Paper: *Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE* — Han Cai et al., NVIDIA, 2026 — https://arxiv.org/abs/2607.07740
- Related: *YaRN: Efficient Context Window Extension of Large Language Models* — Peng et al., 2023 — https://arxiv.org/abs/2309.00071
- Related: *Extending Context Window via Positional Interpolation* — Chen et al., 2023 — https://arxiv.org/abs/2306.15595
