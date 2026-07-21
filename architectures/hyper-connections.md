# Hyper-Connections (HC / mHC / xHC)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** **Hyper-Connections** expand a Transformer's residual stream from one channel into $N$ parallel channels; each block reads and writes across them. It's a third scaling axis beyond width and depth. Vanilla HC saturates at $N{=}4$; the follow-up **mHC** (manifold-constrained HC) stabilizes training at scale; the newest **xHC** (2026) fixes two bottlenecks — insufficient write-back information and cubic-in-$N$ mixing cost — that were preventing meaningful expansion past $N{=}4$.

**Prereqs:** [transformer-block.md](transformer-block.md), [_normalization.md](_normalization.md)
**Related:** [looped-transformer.md](looped-transformer.md)

---

## What it is

Standard residual connections carry a single hidden state through the network — width $d$, one stream. Hyper-Connections split that residual into $N$ streams, each of width $d$ (or a fraction of it). At each block:

- **Read.** The block reads a mixture of the $N$ streams (a learned mixture, typically a small matrix).
- **Compute.** Runs the usual attention / FFN on the mixture.
- **Write.** Distributes its output back across the $N$ streams via a second learned mixture — the "write-back" step.

Between blocks, an inter-stream *residual-mixing* step re-projects across the $N$ streams, letting information flow between them. Both the read-mix and the residual-mixing are trainable.

Empirically, $N{=}1 \to N{=}4$ gives large, cheap gains. Beyond $N{=}4$, gains flatten and training cost blows up.

## How it works

The two bottlenecks xHC identifies:

1. **Insufficient write-back information.** With more streams, each block's write-back has less signal per stream — the same output has to be spread over more channels, so each channel's update becomes noisier. Fix: enrich the write-back path so more information reaches each stream.
2. **Cubic-in-$N$ residual-mixing cost.** Naïve inter-stream mixing is an $N \times N$ matrix multiply over $d$-wide streams — $O(N^2 d)$ per token — plus per-block projection cost that scales cubically with $N$ in the general form. Fix: a factored mixing kernel whose cost scales linearly (or lower) in $N$.

With both fixes, xHC extends the HC family well beyond $N{=}4$ under matched training compute, restoring the "third scaling axis" narrative.

Manifold-Constrained HC (mHC), the predecessor, addresses a separate scaling issue: without a norm constraint on the $N$ streams, one stream can dominate and the others degenerate. mHC projects the stream-vector onto a shared manifold before mixing, keeping streams comparable.

## Why it matters

Width and depth are the classical scaling axes; residual-stream width has become a quiet third one. If HC-style expansion holds beyond $N{=}4$ at practical cost, architects have a genuinely new lever: memory capacity that grows with $N$ streams, decoupled from adding layers or making them wider. Interpretability angle too — the $N$ streams may specialize (some for memory, some for compute), giving cleaner circuits than a single-stream residual.

## Gotchas & tricks

- **Kernel work is required.** Neither mHC nor xHC drops into a stock Transformer library — the read-mix + write-back + residual-mixing need dedicated kernels to be efficient at $N > 4$.
- **Match total-cost, not per-parameter-count comparisons.** HC gains come at extra compute for the mixing steps; report FLOPs-matched baselines.
- **Norm constraint matters.** Without mHC's manifold projection (or an equivalent), you'll see stream-dominance collapse during long training runs.
- **$N$ interacts with head count.** A residual-mixing that ignores attention-head structure can undo per-head specialization. HC-family works cleanly when mixing is done outside the attention block.
- **Not a replacement for depth.** HC adds *parallel* residual capacity; adding layers still buys different, complementary capacity. Use both.

## Sources

- Paper: *xHC: Expanded Hyper-Connections* — Zhang, Qin, Zou, Dai, Shi, Wu, Yang, Xia, Zhang, Yao, Liu, Cheng, Yan — SJTU / Xiaohongshu / USTC / CUHK, 2026 — [arXiv:2607.14530](https://arxiv.org/abs/2607.14530).
- Predecessor: *Hyper-Connections* — the original HC formulation.
- Predecessor: *Manifold-Constrained Hyper-Connections (mHC)* — stability fix used at scale.
