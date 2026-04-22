# DeepSeekMoE
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An MoE design that pushes **expert granularity** (many small experts instead of a few big ones) and adds a **shared always-on expert** that carries cross-domain common knowledge. Each token is routed to top-K of many small experts plus the shared expert. Introduced in the DeepSeekMoE paper (2024), then scaled to 671B total / 37B active in DeepSeek-V3.

**Prereqs:** [_moe](_moe.md), [transformer-block](transformer-block.md)
**Related:** [aux-loss-free-balancing](aux-loss-free-balancing.md)

---

## What it is

Standard MoE designs (Switch, GShard, Mixtral) use a small number of relatively large experts — e.g. 8 experts with the same intermediate size as a dense FFN. DeepSeekMoE argues this is the wrong axis. At fixed **activated parameters per token**, you should:

1. **Subdivide each expert** — split the one big FFN into many small FFNs and route to more of them.
2. **Factor out the shared part** — reserve one or more "shared experts" that are always active, carrying cross-domain knowledge that every token needs, so the routed experts don't waste capacity relearning it.

In DeepSeek-V3 this lands at **256 routed experts + 1 shared expert**, with each routed expert having intermediate dim 2048 (small — compared to a dense model's ~18 000), and **top-8 routed experts per token** plus the always-on shared one. Total params: 671B. Active per token: 37B (of which 8 routed × small + 1 shared + attention + norms).

---

## How it works

### Layer structure

```
            h_t  (token hidden state)
               │
               ├──────────────────► FFN_shared(h_t)           ← always active
               │
               ├─► router ─► top-K routed experts (K=8 of 256)
               │                 │
               │                 ▼
               │          g_{i,t} · FFN_routed_i(h_t)   for each selected i
               │                 │
               └─────────────────┴────────────►   sum  ─► y_t + residual
```

### Routing

Affinity score per routed expert `i`:

```
s_{i,t} = sigmoid( h_t · e_i )          ← e_i is a learned centroid per expert
```

DeepSeekMoE uses **sigmoid per expert** rather than a softmax over all experts. This is a small but deliberate choice: the sigmoid's scores are independent per expert, which means adding a bias term to one expert's score (for aux-loss-free balancing, see [aux-loss-free-balancing](aux-loss-free-balancing.md)) doesn't affect others' scores.

Top-K selection + gating weight normalization:

```
S_t = { top-K_r experts by s_{i,t} }
g_{i,t} = s_{i,t} / Σ_{j ∈ S_t} s_{j,t}      for i ∈ S_t;   0 otherwise
```

### Full FFN output

```
y_t = FFN_shared(h_t) + Σ_{i ∈ S_t} g_{i,t} · FFN_routed_i(h_t)
```

In DeepSeek-V3 the shared expert uses the same FFN shape (intermediate dim 2048) as each routed expert, so its contribution is directly comparable in scale.

### Fine-grained vs coarse: why it works

At fixed active FLOPs per token, which is better: 2 experts of intermediate dim 8192, or 8 experts of intermediate dim 2048? Naively these have the same output space, but the fine-grained version:

- Gives the router **4× more unique combinations** to choose from (`C(8,8)` vs `C(2,2)` — sorry, `C(256,8)` vs `C(8,2)` in the real setting).
- Allows finer-grained specialization — an expert can focus on a narrower slice of data.
- Reduces interference — fewer tokens share each expert, so gradients from unrelated tokens don't fight over the same weights.

The ablations in the DeepSeekMoE paper show that fine-grained experts consistently outperform coarse-grained at matched activated parameters, up to the point where expert count exceeds what the router can reliably distinguish (around 256 experts in practice).

### The shared expert

A single routed expert can't specialize on *common* patterns — if it does, it's selected too often, wasting its capacity. The shared expert decouples this: always active, so it absorbs the cross-domain common knowledge (basic syntax, common English tokens, generic world-knowledge), leaving routed experts free to specialize.

Empirically, removing the shared expert costs 1–2 benchmark points across tasks at equal total FLOPs. Adding it doesn't grow total params meaningfully (one extra FFN out of 257 is a rounding error) but shifts the routing landscape significantly.

### Node-limited routing

At 256 experts spread across many nodes, naive top-K routing can scatter a token's 8 experts across all nodes — cross-node all-to-all cost explodes. DeepSeek-V3 constrains: **each token is routed to experts on at most `M = 4` nodes**. Implemented as a hard mask at the top-K selection step — pick top-M nodes by sum-of-top-scores first, then top-8 experts within those nodes.

---

## Why it matters

- **Better cost-quality Pareto** at large scale than coarse-grained MoE.
- **Specialization you can actually see.** Per-expert activation patterns (shown in the DeepSeekMoE paper) are visibly more specialized in the fine-grained setup — one expert lights up on code, another on math, another on CJK text.
- **Composable with aux-loss-free balancing.** The sigmoid-per-expert routing plays perfectly with per-expert biases for load balancing.
- **Scales to 671B.** This is the MoE recipe that got DeepSeek-V3 to parity with GPT-4o and Claude 3.5 Sonnet at a fraction of the training cost.

---

## Gotchas & tricks

- **Node-limited routing is mandatory at large expert counts.** Without it, all-to-all bandwidth drowns the savings from fine-grained experts. The paper's ablations include this explicitly.
- **Shared expert FLOPs count.** It's always active, so it adds to per-token active FLOPs. Pick the shared expert's intermediate dim deliberately — same as routed-expert dim is a natural choice, but 2× or 0.5× are also defensible. In V3 they chose 1× matching the routed dim.
- **Sigmoid vs softmax.** The sigmoid router is a departure from Mixtral and Switch, which use softmax. Only use sigmoid if you also plan to do aux-loss-free balancing — the biasing trick needs sigmoid's independence-per-expert.
- **Expert capacity is soft.** Unlike GShard's hard capacity factor, DeepSeekMoE doesn't drop tokens — overloaded experts just run on more tokens. Aux-loss-free balancing keeps overloads small in practice.
- **Router parameters are small.** With `d_model = 7168` and 256 experts, the router is `7168 × 256 ≈ 1.8M` parameters per layer — negligible against the ~1.7B of routed FFN params per layer.
- **Training stability sensitive to router init.** Initialize expert centroids `e_i` small so the router is near-uniform at the start; let load balancing push it to specialization over time.
- **Don't confuse with Mixtral's 8 experts.** Mixtral's "8 experts" is coarse-grained (each expert ≈ dense FFN size). DeepSeekMoE's 256 experts each have intermediate dim ~1/8 of that — same total capacity, finer slicing.

---

## Sources

- Paper: *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* — Dai et al., 2024 — the fine-grained + shared expert recipe.
- Paper: *DeepSeek-V2* — DeepSeek, 2024 — first large-scale application.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — 671B total / 37B active configuration.
- Paper: *Mixtral of Experts* — Jiang et al., 2024 — contrasting coarse-grained design.
- Paper: *Switch Transformers* — Fedus et al., 2021 — historical top-1 baseline.
