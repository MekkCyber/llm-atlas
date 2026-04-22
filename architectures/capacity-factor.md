# Expert Capacity Factor
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A hard **cap on the number of tokens each expert can process per step**, expressed as a multiplier on the uniform share. `capacity = (tokens / num_experts) × capacity_factor`. If an expert is selected by more than `capacity` tokens, the **overflow tokens are dropped** — they skip the MoE computation and bypass via the residual connection. Capacity factor trades off wasted FLOPs on padding (higher factor) vs dropped-token quality loss (lower factor). Introduced in GShard (Lepikhin 2020) and stated cleanly in Switch Transformer (Fedus 2021).

**Prereqs:** [_moe](_moe.md)
**Related:** [load-balancing-loss](load-balancing-loss.md) · [deepseek-moe](deepseek-moe.md)

---

## What it is

Sparse-MoE routing produces variable load per expert: even with a load-balance loss, the top-K selection can put more tokens on one expert than another within a given step. But hardware needs **static shapes**: a matmul of shape `[capacity, d_in] × [d_in, d_out]` is fast; a ragged `[variable, d_in] × [d_in, d_out]` is slow or impossible on most accelerators.

Capacity factor resolves this by allocating **a fixed buffer per expert** (the "capacity") and enforcing it at dispatch time. Tokens that would exceed the buffer are dropped — their expert contribution becomes zero — and the token continues through the model via the residual connection. The MoE layer effectively acts as the identity for dropped tokens.

---

## How it works

### The Switch form (clearest statement, Eq. 3)

For a batch with `T` tokens distributed across `N` experts, top-1 routing:

```
expert_capacity  =  ⌈ (T / N) · capacity_factor ⌉
```

Each expert's compute buffer is shaped `[expert_capacity, d_in]`. When the router dispatches tokens to an expert in priority order (first-come, first-served by token index typically), the first `expert_capacity` tokens are processed; any additional tokens hitting that expert overflow.

### The GShard form (per-group, top-2)

GShard uses groups of `S` tokens (batch is partitioned into `G` groups of `S`) and top-2 routing. The per-group, per-expert capacity is:

```
capacity  =  ⌈ 2 · S / E ⌉      (for top-2)
```

The implicit capacity factor in GShard is exactly `2` — just enough to accommodate top-2 routing under uniform load. GShard doesn't expose it as a tunable knob; Switch promotes it to a hyperparameter.

### Overflow handling: drop via residual

Switch §2.2 (direct quote): *"If too many tokens are routed to an expert (referred to later as dropped tokens), computation is skipped and the token representation is passed directly to the next layer through the residual connection."*

Concretely, the MoE combine step zeroes the dropped token's contribution:

```
y_t  =  x_t  +  Σ_{i selected}  g_{i,t} · FFN_i(x_t)
     =  x_t  +  0                                         ← if token t is dropped
     =  x_t
```

GShard does the same. Dropped tokens are **not rerouted to a second-choice expert** — that's a different design (e.g. Expert Choice routing sidesteps the issue entirely).

### Scope: per layer, per batch (or per group)

One capacity buffer **per expert per MoE layer**. Each layer drops independently; a token can be dropped by layer 5 and processed normally by layer 7. Cross-layer drop rates compound — the paper's typical reports are averaged per layer, not across layers.

Switch computes capacity per batch; GShard per group. Neither defines capacity per sequence. In practice, capacity is enforced at whatever granularity the router runs.

### Typical values

From Switch §2.2 and Figure 3, Table 1:

| capacity_factor | Interpretation | Observed behavior |
|---|---|---|
| 1.0 | No slack | Many drops unless router is near-perfect; best wall-clock |
| 1.25 | Light slack | **Switch's default for training** |
| 1.5 | Moderate slack | Fewer drops, ~10–20% more FLOPs wasted as padding |
| 2.0 | Wide slack | Very few drops, significant padding cost |

Direct quote (Switch §2.2): *"Switch Transformers perform better at lower capacity factors (1.0, 1.25)."*

**Train vs eval asymmetry.** ST-MoE §3 (p. 6) states it clearly: *"The train capacity factor is 1.25 and the eval capacity factor is 2.0."* At eval time there's no gradient and no training-signal cost to drops — just quality cost — so paying more FLOPs to avoid drops is the right tradeoff. At train time drops also lose gradient signal, but there's a harder cap on how much padding is tolerable. The 1.25 / 2.0 asymmetric convention is the modern default.

Switch itself does not set different train vs eval factors; ST-MoE is the paper that documents this convention.

### The two failure modes

Capacity factor is a two-sided knob:

- **Too low** → many dropped tokens → noisy gradients at training time, quality degradation at eval.
- **Too high** → each expert's buffer is mostly padding → wasted FLOPs, slower steps.

```
tokens_per_expert_actual  ≤  expert_capacity
padding_fraction          =  (expert_capacity - tokens_actual) / expert_capacity
drop_fraction             =  max(0, tokens_would_route - expert_capacity) / T
```

A well-balanced router with capacity_factor = 1.25 usually sees drop_fraction < 1% and padding_fraction around 10–20%.

---

## Why it matters

- **Makes MoE matmuls hardware-shaped.** Static buffer shapes unlock standard tensor-core paths; without capacity enforcement, all-to-all and the expert matmuls become ragged and slow.
- **The second lever for load balancing.** Aux loss pressures the router statistically; capacity factor sets a hard limit on the *consequences* of imbalance. The two complement: aux loss keeps drops rare, capacity factor absorbs the rare ones.
- **Train / eval asymmetry is cheap quality.** Raising eval capacity to 2.0 avoids drop-induced quality loss at serving time without affecting training.

---

## Gotchas & tricks

- **Drops are not rerouted.** The canonical behavior is: the token's expert contribution is zeroed and it continues via the residual. Some implementations route overflow to a second-choice expert (a "back-off" policy) — that's a non-standard variant; check your framework.
- **Capacity-factor drops do not enter the aux loss.** `f_i` in the load-balance loss counts tokens the router *wanted* to dispatch, not the ones that actually got processed. Drops are invisible to the aux loss.
- **Very low capacity starves small experts.** At capacity_factor = 1.0, any imbalance immediately causes drops. Combined with a poorly-tuned aux loss, training becomes unstable early. Start with 1.25.
- **Eval capacity must be set explicitly.** The model file doesn't carry capacity_factor as a parameter — it's a serving-time config. Silently using the training value (1.25) at eval time costs benchmarks.
- **Per-sequence drops matter more than batch-average drops.** A 5% batch-average drop rate can still mean one sequence has 30% of its tokens dropped. Profile drops at sequence granularity, not just batch average.
- **Capacity counts dispatches, not unique tokens.** For top-K > 1 routing, one token is dispatched K times, each dispatch counts against some expert's capacity. Hence the implicit factor-of-K in GShard's top-2 capacity.
- **FFN capacity vs attention.** Capacity factor applies to MoE FFN layers only. MoE-attention variants (rare) would need their own capacity accounting.
- **Capacity factor is orthogonal to expert count.** Doubling experts at the same capacity factor halves the per-expert capacity — drop rate may go up if the router can't distribute that finely. Fine-grained MoE (many small experts) needs careful capacity-factor retuning.
- **Doesn't apply to Expert Choice routing.** Expert Choice is balanced by construction: each expert picks its top-K tokens, so load is exactly K per expert and capacity factor is moot.
- **Zero overlap with aux-loss-free balancing's cadence.** Aux-loss-free bias updates happen at end of step (after drops are computed). Drops during the step have no influence on the bias update. This is not a bug — it's consistent with the aux-loss design — but worth knowing.

---

## Sources

- Paper: *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* — Lepikhin et al., 2020 — Sec 2.2 "Expert capacity" bullet; implicit capacity factor of 2 for top-2 routing.
- Paper: *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* — Fedus et al., 2021 — Sec 2.2, Eq. 3; introduces capacity_factor as a hyperparameter; Fig. 3 and Table 1 for ablations; quote on 1.0/1.25 being preferred.
- Paper: *ST-MoE* — Zoph et al., 2022 — Sec 3 (p. 6) documents the 1.25 train / 2.0 eval convention.
