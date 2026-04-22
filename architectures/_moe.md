# Mixture-of-Experts (MoE)

*Taxonomy — sparse FFN layers where only a subset of experts is activated per token.*

**TL;DR:** Replace the dense FFN in each transformer block with many smaller **expert** FFNs, and use a lightweight **router** to send each token to only a few experts. Total parameters grow large (capacity), activated parameters stay small (cost). The family breaks into design choices along three axes — **routing algorithm** (top-K token-choice vs expert-choice), **load balancing** (auxiliary loss vs aux-loss-free), and **expert granularity** (few big vs many small, ± shared experts).

**Related taxonomies:** [_normalization](_normalization.md)
**Depth files covered here:** [deepseek-moe](deepseek-moe.md) · [load-balancing-loss](load-balancing-loss.md) · [sequence-wise-balance-loss](sequence-wise-balance-loss.md) · [aux-loss-free-balancing](aux-loss-free-balancing.md) · [capacity-factor](capacity-factor.md)

---

## The problem

Dense transformers pay for every parameter on every token. Beyond ~70B active params, FLOP cost per token becomes prohibitive for training and serving. But most of a model's knowledge is *specialized*: a French-cooking token and a C++-header token don't need the same parameters. MoE lets you grow total parameters (more knowledge) without growing active parameters (same FLOPs per token) by routing each token to a specialized subset.

What goes wrong if you do this naively:
- **Router collapse** → all tokens go to one or two experts; the rest die.
- **Communication cost** → cross-device token shuffling becomes the bottleneck.
- **Training instability** → router's discrete choices make gradients noisy; auxiliary losses interfere with the main objective.

Every MoE variant is a different set of answers to these three problems.

---

## The shared pattern

```
h_t  ──▶  router(h_t)  ──▶  top-K_r expert indices, gating weights g_{i,t}
                                      │
                                      ▼
              y_t = Σ  g_{i,t} · FFN_i(h_t)          (+ FFN_shared(h_t) if used)
                 i ∈ Top-K
```

Every MoE variant has:
- A **router** (usually a single linear map `h · e_i` or similar affinity score) that picks which experts to use per token.
- **Top-K_r routing**: each token is processed by `K_r` experts out of `N` total (`K_r = 1, 2, 4, 8, ...`).
- A **load-balancing mechanism** so experts stay roughly equally used.
- Usually an **all-to-all** communication step to shuffle tokens to their assigned experts' devices and back.

---

## Variants

MoE design breaks into two kinds of entries: **system-level designs** (whole models / routing regimes) and **mechanism-level techniques** (specific ideas used inside those designs). This table lists mechanism-level techniques — whole-model designs like Switch, GShard, and Mixtral belong in [case-studies/](../case-studies/) when they get written up.

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [**DeepSeekMoE**](deepseek-moe.md) (Dai 2024) | Fine-grained (many small experts) + shared always-on expert | More experts = bigger all-to-all cost | Maximum specialization at given activated-FLOPs budget |
| [**Load-balancing auxiliary loss**](load-balancing-loss.md) (GShard 2020, Switch 2021) | Differentiable surrogate `α·N·Σ_i f_i·P_i` for uniform expert usage | Gradient interferes with main task loss | Default mechanism; simple, well-understood baseline |
| [**Sequence-wise balance loss**](sequence-wise-balance-loss.md) (DeepSeekMoE 2024, DeepSeek-V3 2024) | Same form, but per-sequence statistics | Higher variance; can hurt within-domain specialization at large α | When single-sequence concentration matters (fine-grained MoE) |
| [**Aux-loss-free balancing**](aux-loss-free-balancing.md) (DeepSeek 2024) | Per-expert bias added to router scores, updated by control loop (no gradient) | Requires tracking per-expert load; extra bias state | Avoids auxiliary-loss gradient interference; frontier default |
| [**Capacity factor + token dropping**](capacity-factor.md) (GShard 2020, Switch 2021) | Hard cap `⌈c_f · T/N⌉` on tokens per expert; overflow bypasses via residual | Padding waste (high `c_f`) vs drop-induced quality loss (low `c_f`) | Universal — needed for static-shape matmuls regardless of balancer |
| **Expert Choice routing** (Zhou 2022) — *no depth file yet* | Experts pick their top-K tokens (not the other way around) | Some tokens routed 0×, others K× | Perfect load balance by construction; good for training |
| **BASE Layers** (Lewis 2021) — *no depth file yet* | Assignment via linear assignment (balanced by construction) | Solver cost; harder to scale | Research / middle scale |
| **Router z-loss** (ST-MoE, Zoph 2022) — *no depth file yet* | Penalize `(log Σ_j exp(logit_j))²` to bound router-logit magnitude | Another aux term (mild gradient interference) | Stability at large scale; complements load-balance loss |
| **Node-limited routing** (DeepSeek-V3, 2024) — *no depth file yet* | Hard cap on nodes a token's top-K can span | Reduces routing flexibility at large expert counts | Fine-grained MoE with constrained inter-node bandwidth |

---

## How to choose

**Default for new large-scale pretraining (2025+):** DeepSeekMoE-style fine-grained + shared experts, with aux-loss-free balancing and top-8 routing. DeepSeek-V3 and several follow-ons have validated this recipe at the 500B+ total parameter scale.

**If you want the simplest thing that works:** Mixtral-style 8 experts, top-2, aux loss. Less specialization, but minimal moving parts and well-understood training dynamics.

**If compute-bound on all-to-all:** fewer experts with coarser granularity, or Expert Choice routing. Fine-grained MoE only wins if your interconnect can absorb the all-to-all cost.

**For inference-only / post-training quantization:** MoE doesn't play specially well with weight-only quantization schemes — each expert is quantized independently, which works but loses some of the dense-layer co-quantization tricks. This is an active area.

### Routing algorithm cheat sheet

- **Top-K token-choice** (standard): each token picks its K favorite experts. Simple, causes load imbalance.
- **Top-K expert-choice**: each expert picks its K favorite tokens. Perfectly balanced. Tokens may be over/under-selected. Works well for training.
- **Hash / random routing**: router is not learned, just a hash. Trivially balanced, but throws away specialization. Baseline / ablation.

### Load balancing cheat sheet

- **[Auxiliary loss (batch-wise)](load-balancing-loss.md)** (Switch, GShard, Mixtral): `α · N · Σ_i f_i · P_i`, computed per MoE layer over the full batch. Simplest mechanism, well-understood, interferes with task gradients.
- **[Sequence-wise balance loss](sequence-wise-balance-loss.md)** (DeepSeekMoE, DeepSeek-V3): same functional form, but `f_i` and `P_i` computed per sequence. Stricter constraint — prevents single-sequence expert concentration.
- **[Aux-loss-free](aux-loss-free-balancing.md)** (DeepSeek-V3): control-loop bias on router scores, no gradient signal. Works better at scale than classical aux loss.
- **Expert Choice**: balance by construction (each expert picks its top-K tokens). No loss term needed, but changes routing semantics — some tokens may be unrouted while others over-routed. (No depth file yet.)
- **[Capacity factor + token dropping](capacity-factor.md)** (GShard, Switch): cap each expert at `⌈capacity_factor · tokens/E⌉` tokens. Overflow tokens are **dropped — they bypass the MoE layer via the residual connection**, not rerouted. Limits imbalance damage at the cost of dropped-token quality loss.

### Expert granularity cheat sheet

- **Few big experts** (Mixtral: 8 × large FFN): each expert has enough capacity to specialize meaningfully, but the router has few choices.
- **Many small experts** (DeepSeekMoE: 256 × small FFN, top-8): router picks a finer combination; each token effectively synthesizes its FFN from a basis.
- **+ shared expert**: one always-on expert carrying cross-domain common knowledge, so routed experts don't waste capacity relearning it.

---

## Adjacent but distinct

- **Conditional computation in general** (early-exit, dynamic depth) — also activates parameters selectively but not by expert specialization. Different category.
- **LoRA / adapters** — parameter-efficient adaptation, not sparse activation. An adapter is always active; an MoE expert is sometimes active.
- **Hypernetworks** — generate weights conditionally. More extreme than MoE and rarely used at LLM scale.

---

## Sources

- Paper: *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* — Shazeer et al., 2017 — the MoE-in-transformers ancestor.
- Paper: *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* — Lepikhin et al., 2020.
- Paper: *Switch Transformers* — Fedus et al., 2021 — top-1 sparse routing, auxiliary load-balance loss.
- Paper: *ST-MoE: Designing Stable and Transferable Sparse Expert Models* — Zoph et al., 2022 — router z-loss, stability recipe.
- Paper: *Mixtral of Experts* — Jiang et al., 2024 — open-source 8x7B / 8x22B.
- Paper: *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* — Dai et al., 2024 — fine-grained + shared experts.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — aux-loss-free balancing at 671B scale.
- Paper: *Mixture-of-Experts with Expert Choice Routing* — Zhou et al., 2022.
