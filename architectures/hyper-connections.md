# Hyper-Connections
*Depth — multi-stream residual connections and their tensor-parallel variants.*

**TL;DR:** Replace the single residual stream in a transformer with `n` parallel streams; each sublayer reads a learned mixture of streams and its output is dispatched back to (one or many of) them. Gives extra capacity *without* new parameters per sublayer. **TPHC** turns the tensor-parallel shard boundary into the stream grouping, making multi-stream residuals essentially free when you already pay for TP.

**Prereqs:** [transformer-block](transformer-block.md)
**Related:** [_normalization](_normalization.md), [../pre-training/_training-stability](../pre-training/_training-stability.md), [multi-branch-transformer](multi-branch-transformer.md)

---

## What it is

A generalization of the residual stream. A transformer block normally has one residual: `h ← h + f(h)`. Hyper-connections (HC) carry `n` streams `h_1,…,h_n`; each sublayer `f` reads a mixture, updates one or more streams, and runs normalization against its chosen read mixture. **mHC** (manifold-constrained HC) adds constraints so streams stay well-conditioned. **TPHC** pairs streams with tensor-parallel shards.

## How it works

With `n` streams, each sublayer gets a *read-weight* vector `r ∈ R^n` and *write-weight* vector `w ∈ R^n`, both learned or scheduled:

```
input   = Σ_i r_i · h_i
output  = f(norm(input))
h_i    ← h_i + w_i · output   for i = 1..n
```

**mHC** parameterizes `r, w` so the resulting multi-stream dynamics stay bounded (manifold constraint on the (n×n) read-write matrix), which is what lets training stay stable at depth.

**TPHC** aligns streams with tensor-parallel groups. In vanilla TP, a sublayer is sharded across `g` GPUs and the output is **all-reduced** into one tensor, then added to *every* stream. TPHC does **no all-reduce**: shard `j`'s output is written only into stream `j`. Each stream receives a different fraction of the compute and develops different features, at zero extra parameter cost.

```
# Vanilla mHC                      # TPHC
all_reduce(output_j across j)       keep output_j local
∀i: h_i += w_i · reduced_output     h_j += w · output_j   (per-stream write)
```

## Why it matters

- **Free capacity.** TPHC gets stream specialization without buying new parameters: the TP shard split, which already exists, becomes the diversity source.
- **Training stability at depth.** Multiple residual pathways reduce gradient attenuation through very deep stacks; mHC's manifold constraint prevents the usual multi-residual blow-up.
- **Adoption.** Used in Qwen variants and the Xing 4 29B model (as **mHC**) and at TP-boundary scale in TPHC. The pattern is quietly becoming default for ≥1B models.

Reported numbers: TPHC lowers validation loss vs. mHC on 1.1B models at matched budget; overhead is ~9% training time at four streams on 8B models.

## Gotchas & tricks

- **All-reduce removal is the whole point of TPHC.** If your runtime still all-reduces across TP shards you get vanilla mHC, not TPHC. Audit comm kernels, not just the config.
- **Stream count tradeoff.** `n=2` is enough for stability wins; `n=4` buys most of the capacity win in TPHC; beyond `n=8` returns diminish and schedule complexity grows.
- **Normalization is per-read mixture.** Each sublayer's LayerNorm/RMSNorm is applied to `Σ_i r_i · h_i`, not to any single `h_i`. Mis-wiring this produces NaNs at init.
- **Does not compose freely with pipeline parallelism.** PP boundaries split the residual stream in time — multi-stream state across a PP boundary needs careful handling. HSDP (no PP) sidesteps the issue.

## Sources

- Paper: *TPHC: Improving Hyper-Connections with Tensor-Parallel Grouped Updates* — Hu, Wu, Wu, Zhang, Xie, Tu (ShanghaiTech / Meituan), 2026 — [arXiv:2611.23026](https://arxiv.org/abs/2611.23026)
- Earlier primary source for HC / mHC: hyper-connections line in the DeepSeek / Qwen literature (2024–2025).
