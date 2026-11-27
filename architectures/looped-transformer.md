# Looped Transformer
*Depth — reuse the same block weights across multiple forward passes, sharing KV so looping does not inflate inference cost.*

**TL;DR:** A transformer in which a *subset* of blocks (often the middle stack) is applied repeatedly instead of being duplicated with separate weights. The parameter count stays small but effective depth grows. The 2026 **Mixture-of-Loops Transformer (MoLT)** adds the critical twist that makes looping competitive at inference time: the KV cache is **shared across loop iterations and across the U-shaped depth stack**, so an $N$-loop forward pass does not produce $N\times$ the KV state.

**Prereqs:** [transformer-block](transformer-block.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [_moe](_moe.md) · [looped-dit](looped-dit.md) · [../inference/README.md](../inference/README.md)

---

## What it is

A classic transformer is a stack of $L$ blocks, each with its own weights. A looped transformer reuses the weights of a single block (or a small stack of blocks) across $N$ iterations, so the effective depth is $N \cdot k$ but the parameter count is only $k$ blocks' worth. The idea predates LLMs (ALBERT, Universal Transformer) but suffered from a persistent problem: at inference time, $N$ passes meant $N \times$ latency, erasing the capacity-efficiency win.

**MoLT's contribution** is to pair the loop with aggressive KV reuse. In MoLT, the attention KV state computed on loop $n$ is **not recomputed** on loop $n+1$ — it is read back from the cache, so successive loops add FFN cost but not fresh attention cost. The U-shaped variant (U-MoLT) extends this across the depth hierarchy.

## How it works

The forward pass over the looped middle stack is:

```
x = outer_pre_blocks(x)            # a few non-looped entry blocks
kv_cache = {}
for n in range(N_loops):
    for block in middle_stack:     # same block weights every loop
        q = Wq(x); k = Wk(x); v = Wv(x)
        if n == 0:
            kv_cache[block] = (k, v)
        else:
            k, v = kv_cache[block]   # ← KV sharing across loops
        x = attention(q, k, v) + x
        x = FFN(x) + x
x = outer_post_blocks(x)
```

Two levers set effective capacity: the loop count $N$ (depth lever) and the number of distinct blocks per loop (width-of-block lever). U-MoLT further organizes the loop over a U-shaped set of blocks (coarser outer, finer middle), with KV also shared across the U's symmetric levels.

## Why it matters

- **Parameter-efficient scaling.** At 700M parameters, U-MoLT reports **1.7× token-efficiency** on validation cross-entropy vs. a non-looped OLMo3 baseline — i.e. you reach the same loss with 60% of the tokens.
- **Looping is free at inference.** With shared KV, an S-size (~113M) U-MoLT matches the inference latency of a non-looped OLMo3 of equivalent capacity. This is the result that reopens looped architectures after years of latency-killed attempts.
- **Composes with MoE and sparsity.** Loops add depth; experts add width. [looped-dit](looped-dit.md) combines both in the diffusion-transformer setting.

## Gotchas & tricks

- **KV sharing changes gradient flow.** If the loop reuses KV but the FFN rewrites $x$ each iteration, the attention logits on later loops are computed with "stale" keys relative to current queries. In MoLT this is deliberate — later loops refine the representation against a fixed memory — but naïve weight sharing *without* KV reuse gets a different (and usually worse) model.
- **Not every block should loop.** The paper keeps a small entry/exit stack of non-looped blocks; looping the entire network tends to under-use the parameters at the extremes.
- **Loop count can be made adaptive.** For diffusion backbones (see [looped-dit](looped-dit.md)), the loop count is scheduled per noise level; for LLMs, an adaptive loop count per sequence is a natural follow-up.
- **Cache size is bounded, not grown, by loops.** The key efficiency gain is that KV cache size scales with the sequence length (as usual), not with the loop count — otherwise the trick would blow up long-context memory.

## Sources

- Paper: *Mixture-of-Loops Transformer* — Zhu, Wang, Wang, Bai, Wang, Bai, 2026, arXiv 2611.34923 — KV-shared looped transformer; the U-MoLT variant.
- Related: *ALBERT* — Lan et al., 2019 — early parameter-shared transformer.
- Related: *Universal Transformers* — Dehghani et al., 2018 — recurrent looping over a single transformer block.
