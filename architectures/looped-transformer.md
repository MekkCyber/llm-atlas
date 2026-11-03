# Looped Transformer (Weight-Tied Depth)
*Depth — apply the same transformer stack twice (or more) to the same tokens, trading latency for effective depth at constant parameter count.*

**TL;DR:** A standard transformer has $L$ unique blocks and depth $L$. A **looped** transformer has $L$ unique blocks applied $N$ times: effective depth $L \cdot N$, storage still $L$. Weight-tying across loop iterations means parameters don't grow, but FLOPs and latency do. Metis-1.6 (Lernex, 2026) applies this at $N{=}2$ ("think twice") on top of a hybrid Mamba-2 + attention MoE backbone, trading inference latency for better per-token quality at fixed storage. The construction generalizes earlier Universal Transformer / Albert-style weight-tying.

**Prereqs:** [transformer-block.md](transformer-block.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [_moe.md](_moe.md) · [deepseek-moe.md](deepseek-moe.md) · [../case-studies/metis-1-6.md](../case-studies/metis-1-6.md)

---

## What it is

A standard transformer forward pass is a composition of $L$ *distinct* blocks:

$$
y = f_L \circ f_{L-1} \circ \cdots \circ f_1 (x)
$$

Each $f_\ell$ has its own weights. Total parameter count scales with $L$.

A looped transformer uses $L$ unique blocks but applies the whole stack $N$ times:

$$
y \;=\; (\,f_L \circ \cdots \circ f_1\,)^{(N)}(x)
$$

Weight-tied across the $N$ iterations. Effective depth $L \cdot N$; storage $L$.

Metis-1.6 is the canonical modern instance at $N{=}2$. The term "think twice" in Lernex's report refers specifically to this double application.

## How it works

### The loop

```
h ← x
for n in 1..N:
    for ℓ in 1..L:
        h ← f_ℓ(h)
return h
```

With $N{=}2$ this is literally two full forward passes through the same stack, taking the previous pass's output as the next pass's input.

### Weight-tying semantics

All $N$ iterations share one set of weights. Backprop sums gradients from each iteration:

$$
\nabla_\theta L \;=\; \sum_{n=1}^{N} \frac{\partial L}{\partial f^{(n)}} \,\frac{\partial f^{(n)}}{\partial \theta}
$$

Gradient magnitude per unique parameter scales with $N$ — effective learning rate implicitly grows. Papers typically compensate with a smaller base LR or a norm-based scaling.

### What loops are *not*

- **Not deeper weights.** Depth$=L \cdot N$, but only $L$ unique transforms.
- **Not recurrence over tokens.** The loop is over *depth*, not sequence position.
- **Not test-time inference scaling in the Monte-Carlo sense.** The number of loop iterations is fixed at train time (though in principle could be increased at inference; Metis-1.6 does not do this).

### Where it fits in a stack

In Metis-1.6, the loop wraps the whole hybrid Mamba-2 + attention MoE stack. In Universal Transformer (Dehghani et al., 2018), the loop is over an attention + FFN block with an adaptive-computation-time halting mechanism. ALBERT (Lan et al., 2019) ties weights across all layers (equivalent to extreme sharing without an explicit outer loop count).

## Why it matters

- **Depth-for-free at inference-latency cost.** Looping doubles effective depth with no extra storage. Useful when memory is the binding constraint (small-model MoE: Praxis has 2.94B total but only 0.46B active, so storage is already dear).
- **Compute budget shifting.** Prefer latency over parameters when serving many concurrent small-model requests on memory-bound hardware.
- **Cheap test-bed for "think longer by iterating".** The loop is a hard structural prior for iterated computation, which some recent work argues is complementary to CoT.
- **Simple drop-in** on existing transformer code: wrap the forward pass in a loop; adjust LR; retrain.

## Gotchas & tricks

- **Effective learning rate inflates with $N$.** Each unique parameter receives $N$ gradient contributions per step. Scale base LR by roughly $1/\sqrt{N}$ or add gradient clipping tuned for the inflated magnitude.
- **No per-iteration adaptation.** Fixed $N$ means the model can't skip work on easy tokens. Universal Transformer's ACT halting addresses this; Metis-1.6 does not.
- **Residual-stream dynamics shift.** Each loop iteration adds a full stack's worth of updates to the residual stream, so the stream's norm trajectory is roughly $N$× what an unlooped model would see. Watch out for norm blowup; adjust init / scale.
- **Interaction with KV cache.** Serving a looped model needs careful cache design — do the attention layers reuse KV across loop iterations or recompute? Metis-1.6's report is thin on this.
- **Latency doubles.** Not literally $2\times$ (overhead amortizes), but close. For latency-sensitive serving this is often disqualifying.
- **Can regress at small model sizes.** Looping below some size threshold has historically hurt (Albert underperformed BERT at BERT-base scale). Metis's claim is that at modern MoE scales + hybrid backbones the regression doesn't appear; broader confirmation pending.

## Sources

- Paper: *Think Twice: The Metis-1.6 Technical Report* — Lernex (single researcher), 2026 — arXiv 2610.37903. Modern instance at $N{=}2$ on hybrid Mamba-2 + attention MoE.
- Paper: *Universal Transformers* — Dehghani et al., 2018 — the first weight-tied looped transformer with adaptive halting.
- Paper: *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations* — Lan et al., 2019 — cross-layer weight tying.
- Related: [deepseek-moe](deepseek-moe.md) — the MoE routing stack Metis sits on top of.
