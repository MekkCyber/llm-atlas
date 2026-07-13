# Jet-Long (Dynamic Bifocal RoPE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A tuning-free zero-shot long-context extension for RoPE-based LLMs that runs **two attention windows in parallel** — a local window at the original RoPE frequency, and a long-range window whose rescaling factor **adapts to the current sequence length**. An inclusion–exclusion merge combines them with no double-counting; an on-the-fly correction rotation keeps the two views consistent. Fused into a single CuTe kernel, long-context prefill reaches up to **1.39× FlashAttention-2 throughput on H100**, and single-batch generation adds ≤ 4% overhead at every length. Introduced by Cai (NVIDIA), 2026 (arXiv 2607.07740).

**Prereqs:** [rope.md](./rope.md), [attention.md](./attention.md)
**Related:** [dca.md](./dca.md) · [_positional-encoding.md](./_positional-encoding.md) · [multi-head-attention.md](./../architectures/multi-head-attention.md)

---

## What it is

A zero-shot RoPE-extension method that removes the "single rescaling factor" tradeoff at load time. Standard methods (PI, NTK-aware, YaRN, ABF) pick one factor: aggressive factors extrapolate but sacrifice short-context fidelity; conservative factors preserve short-context but break at long. Jet-Long ships **both**, in parallel, and merges them so the model behaves like its base checkpoint on short inputs and like a rescaled long-context model on long ones — without any fine-tuning.

## How it works

Two attention windows share the same Q/K/V but see different position views.

**Local window (RoPE-faithful):** The first $w$ tokens attend under the pretrained RoPE frequency — no rescaling. Behaves exactly like the base model.

**Long-range window (dynamic-rescale):** The remaining prefix uses a rescaling factor $s(L)$ that is a function of the *current* input length $L$. Concretely, $s(L)$ is chosen so the effective RoPE base at length $L$ interpolates from 1× (short) to a length-appropriate scale-up (long). This is the "dynamic" in dynamic bifocal.

**Correction rotation:** Q and K enter the two windows under two RoPE bases. An **on-the-fly correction rotation** applied before the long-range window realigns them so the same Q/K pair is scored consistently.

**Inclusion–exclusion merge:** Tokens covered by both windows would be double-counted in a naive union. Jet-Long merges the two softmax outputs via an inclusion–exclusion formula in log-space, giving a well-formed attention distribution over the full prefix without an extra normalizing pass.

**Fused kernel:** Because all three pieces (local softmax, long-range softmax, correction rotation, merge) are analytic, the whole thing lowers to a single CuTe kernel. The paper reports up to 1.39× FA2 throughput on long-prefill on H100, approaching the Hopper-only FA4 numbers.

## Why it matters

Frontier deployments extend context zero-shot — retraining a 70B+ model on a 128K corpus is expensive and rarely done for every new context target. Every existing zero-shot method forces a load-time compromise: one rescaling factor for the whole run. Jet-Long collapses that compromise into a single artifact that serves short and long simultaneously, and beats FA2 rather than costing throughput. Concretely, it changes the "which RoPE-extension do I ship?" decision from "pick one" to "just use Jet-Long."

## Gotchas & tricks

- **Dynamic factor at run time.** $s(L)$ is computed per input. The merge is analytic but the rescaling schedule is the load-bearing hyperparameter — the paper's schedule is tuned against Llama-family bases and may need re-tuning for other families.
- **CuTe kernel is Hopper-optimized.** The 1.39× number is on H100. On A100/Ampere the kernel exists but the throughput advantage over FA2 shrinks.
- **Composes with existing schemes.** Jet-Long is a bifocal wrapper over RoPE — the long-range window can itself use ABF, YaRN, or NTK as its rescaling family. The paper's default is a YaRN-flavored rescale.
- **Short-context recovery is exact.** At small $L$, $s(L) \to 1$ and the long-range window collapses into the local window; the model behaves identically to the base. Different from PI/NTK/YaRN which degrade short-context quality by construction.

## Sources

- Paper: *Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE* — Han Cai (NVIDIA), 2026 — [arXiv 2607.07740](https://arxiv.org/abs/2607.07740).
- Related paper: *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al., 2021.
- Related paper: *YaRN: Efficient Context Window Extension of Large Language Models* — Peng et al., 2023.
- Related paper: *Training-Free Long-Context Scaling of Large Language Models* (DCA) — An et al., 2024.
