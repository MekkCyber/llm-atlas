# Meta Expert

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An MoE variant that **shares a single global expert pool across all transformer layers** instead of giving each MoE layer its own private pool. At matched activated experts and granularity, Meta-Expert MoE lowers cross-entropy and raises downstream accuracy while using only ~70–80% of the experts that vanilla MoE needs for the same quality.

**Prereqs:** [_moe](_moe.md), [deepseek-moe](deepseek-moe.md), [transformer-block](transformer-block.md)
**Related:** [aux-loss-free-balancing](aux-loss-free-balancing.md), [capacity-factor](capacity-factor.md)

---

## What it is

Every mainstream MoE (Switch, GShard, Mixtral, DeepSeekMoE) instantiates a **private** expert pool at every MoE layer. The authors of Meta Expert argue this pays twice: once in storage (private pools balloon total params with cross-layer-duplicated computations) and once in retrieval (routing + all-to-all scales with per-layer pool size). They propose collapsing all per-layer pools into one **shared "meta" pool** that every layer routes into.

A token's hidden state at each layer is routed to top-$K$ experts from the *same* pool, with the layer identity injected as a conditioning signal (so a given expert can behave differently when invoked from different depths).

---

## How it works

### Layer structure

```
                     Meta Expert pool  { E_1, E_2, ..., E_N }
                                        ▲
                                        │  (one shared bank)
  Layer ℓ:   h^ℓ_t ──► router_ℓ ─► top-K indices + gates
                                   │
                                   ▼
                         y^ℓ_t = Σ g_{i,t} · E_i( h^ℓ_t ; ℓ )     (ℓ = layer id)
```

- One global pool of $N$ experts shared across all MoE layers.
- Each layer still has its own router (so routing policy can specialize by depth).
- Expert input is conditioned on layer identity ℓ (via a small layer-embedding) so the same expert function can play different roles at different depths.
- Activated-experts-per-token $K$ is held constant — only the pool is deduplicated across depth.

### Why depth-shared experts don't collapse

The counter-intuitive part: if the same expert is called at multiple depths, won't it have to average those roles? The paper's argument is that **the residual stream reshapes the hidden state between layers**, so the same expert receives *different distributions* of inputs at different depths. Shared experts become generic transformations applied to depth-specific inputs, enriching the residual pathway rather than collapsing capacity.

### Comparison with Mixture-of-Depths and shared experts

- **Shared expert** (DeepSeekMoE): one always-on expert per layer, still *per-layer* parameters. Meta Expert shares the whole pool.
- **Cross-layer parameter sharing** (ALBERT-style): shares *all* parameters across layers, including attention and dense FFN. Meta Expert shares only the expert bank, not the router or attention.

---

## Why it matters

- **Storage saving at matched quality.** Meta Expert matches vanilla MoE with ~70–80% of the experts. Total-parameter savings compound into expert-parallel memory and all-to-all bandwidth savings.
- **Downstream accuracy up.** Lower cross-entropy and higher accuracy at matched activation budget in the paper's controlled comparisons.
- **Clean composability.** The modification is orthogonal to balancing (aux-loss-free, aux-loss, expert-choice) and to granularity (fine-grained + shared). Can be stacked onto DeepSeekMoE-style recipes.
- **Theoretical pathway story.** Depth-shared experts give the residual stream more flexible paths across layers (any expert reachable at any depth), which the paper argues improves information retrieval.

---

## Gotchas & tricks

- **Layer-id conditioning is load-bearing.** Without it, shared experts do collapse — the ablation shows most of the gain comes from letting the same expert specialize *conditioned* on depth, not from the sharing alone.
- **Expert-parallel sharding changes.** When experts are shared across layers, naive expert-parallel sharding breaks the usual assumption that an expert lives at one place in the compute graph. Collective-comm patterns need to be redesigned.
- **Router capacity grows.** Each layer still has its own router over the full pool, so router params actually grow (not shrink). Partially offsets the expert-pool savings — net storage saving is dominated by shrinking the pool, not the routers.
- **Granularity interaction unclear.** The paper runs controlled comparisons at fixed activated experts and granularity. Whether the 70–80% ratio holds at extreme fine-grained ratios (DeepSeekMoE's 256+) or at coarse Mixtral-8-style setups needs separate validation.

---

## Sources

- Paper: *Meta Expert: Cross-Layer Expert Sharing for Storage- and Retrieval-Efficient MoE* — Hong et al., Tsinghua / Shanghai AI Laboratory, 2026 — [arXiv:2610.27598](https://arxiv.org/abs/2610.27598).
- Paper: *DeepSeekMoE* — Dai et al., 2024 — fine-grained-expert ancestor whose per-layer pool Meta Expert replaces.
- Paper: *ALBERT* — Lan et al., 2019 — the cross-layer-shared-parameters precedent in dense models.
