# RoPE Post-Hoc Context Extension

*Taxonomy — techniques for extending a trained RoPE model's context length after pretraining.*

**TL;DR:** A model trained at context length $L$ can be pushed to $L' \gg L$ by rescaling RoPE's rotation angles rather than retraining from scratch. The methods differ on which frequencies get rescaled, whether they touch positions or the base, and how much fine-tuning they need. Modern deployment stacks default to **YaRN** for high-quality fine-tuned extension and **ABF** or **Jet-Long** for tuning-free.

**Related taxonomies:** [_positional-encoding.md](_positional-encoding.md)
**Depth files covered here:** [rope](rope.md) · [jet-long](jet-long.md)

---

## The problem

RoPE encodes position by rotating $q$ and $k$ by $m \cdot \theta_i$ per token. At inference on a sequence longer than the training length, the rotation angles $m\theta_i$ exceed anything the model has seen — attention scores explode, perplexity blows up. The community's discovery: rather than retrain, *scale the RoPE frequencies* so that the effective rotation range stays within the trained regime.

## The shared pattern

Every method modifies the RoPE frequency spectrum $\{\theta_i\}$ or the position index $m$ so the *effective* angle stays inside the training window. They differ on three axes:

1. **What is rescaled** — positions ($m \to m/s$), the base ($b \to b'$), or a per-dimension mix.
2. **When it applies** — uniformly across dimensions (PI, ABF), only long-wavelength dimensions (YaRN, LongRoPE), or length-adaptively (Jet-Long).
3. **Fine-tune cost** — full retrain, short adapter, or tuning-free.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Position Interpolation (PI) | Divide positions by $s = L'/L$ before RoPE | Uniform squish loses high-frequency content | Simple; needs short fine-tune (~1k steps) |
| NTK-aware scaling | Scale base $b \to b \cdot s^{d/(d-2)}$ | Uniform base scaling — subtler than PI | Tuning-free at moderate extension |
| YaRN | NTK-by-parts + attention temperature | Requires short fine-tune | **SOTA quality** for 8k → 128k with light fine-tune |
| Base-frequency (ABF) | Larger fixed base (500k / 1M) | Single hyperparam; no adaptivity | Cheap, one-line config change (Llama 3, Kimi k1.5) |
| LongRoPE (no depth file yet) | Non-uniform per-dim scaling from search | Search cost | Aggressive extension (>1M tokens) |
| [Jet-Long](jet-long.md) | Bifocal RoPE, length-adaptive long-range factor | 2× RoPE compute, small constant | **Tuning-free**, no short-context regression |

## How to choose

**If you can afford a short fine-tune**, YaRN is the default — best quality-per-token, mature ecosystem support. **If tuning-free**, ABF is the simplest one-line change; **Jet-Long** improves on ABF whenever short-context fidelity must be preserved exactly (agent stacks, RAG that mixes short and long prompts). **NTK-aware** and **PI** are mostly historical baselines now, though NTK-aware still shows up in older codepaths.

**Combine with a long-context fine-tune** for maximum quality at extreme lengths — the tuning-free methods hit a soft ceiling that a supervised long-context stage clears.

## Adjacent but distinct

- **Attention variants** (MLA, GQA, linear attention). Orthogonal — pick each independently. MLA notably keeps a decoupled RoPE slice so all these extension methods still apply.
- **DCA (Dual Chunk Attention).** Chunked attention with modified position indices — different family from frequency-rescaling; see [dca.md](dca.md).
- **Training-length schedules.** Gradually growing context during pretraining or mid-training is a separate topic.

## Sources

- Paper: *Position Interpolation* — Chen et al., 2023 — [arXiv:2306.15595](https://arxiv.org/abs/2306.15595).
- Paper: *YaRN* — Peng et al., 2023 — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071).
- Paper: *Scaling Laws of RoPE-based Extrapolation* — Liu et al., 2023 — [arXiv:2310.05209](https://arxiv.org/abs/2310.05209).
- Paper: *LongRoPE* — Ding et al., 2024.
- Paper: *Jet-Long* — Cai et al., NVIDIA, 2026 — [arXiv:2607.07740](https://arxiv.org/abs/2607.07740).
