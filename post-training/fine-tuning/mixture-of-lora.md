# Mixture-of-LoRA
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of one adapter per task, keep a growing library of LoRA adapters over a shared frozen base and add a **router** that picks (or blends) which adapters activate for the current input. New skills are added by minting new adapters — the base model never drifts and old skills don't get overwritten, so continual learning becomes an *adapter-graph growth* problem rather than a *weight-update stability* problem.

**Prereqs:** [lora.md](lora.md), [../../architectures/_moe.md](../../architectures/_moe.md)
**Related:** [../_post-training.md](../_post-training.md), [../../architectures/load-balancing-loss.md](../../architectures/load-balancing-loss.md), [../../architectures/aux-loss-free-balancing.md](../../architectures/aux-loss-free-balancing.md)

---

## What it is

Standard LoRA gives you *one* frozen base plus *one* adapter per task — switching tasks means reloading the adapter. Mixture-of-LoRA (MoL) turns that library into an active mixture: at each layer, a learned router scores the current token/prompt against every adapter's activation gate and either selects the top-k or produces a soft blend. Popularized in 2026-era open agent families (e.g. Macaron-V1) as the substrate for continual learning.

Structurally close to standard MoE, but the "experts" are rank-`r` LoRA updates over a shared base weight rather than independent FFN blocks. This keeps the parameter budget tiny compared to full MoE — each adapter is `2rd` parameters, not the `4d²` of an FFN expert — while still delivering the routing-based specialization of MoE.

## How it works

For a target linear layer with frozen weight `W` and adapters `{B_i A_i}` for `i = 1..N`:

```
gate = softmax( router(x) )          # gate ∈ R^N, router is a small linear head
active = topk(gate, k)               # k typically 1–4
h = W·x + Σ_{i ∈ active} gate_i · (α/r) · B_i · A_i · x
```

Router is trained jointly with the adapter weights. New adapters can be minted at any time — either by training a new (`A`, `B`) pair on a new task's data with the router extended and the old adapters frozen, or by letting the router itself decide when to allocate a new adapter (recursive self-improvement).

Load balancing typically borrows from MoE: either an auxiliary loss encouraging uniform adapter utilization, or a bias-based aux-loss-free scheme like DeepSeek-V3's — same problem, same fixes.

## Why it matters

- **Continual learning without catastrophic forgetting.** Because the base is frozen and old adapters are never updated, adding a new skill cannot destroy an old one. The failure mode moves from *drift* to *routing interference*, which is a much more tractable problem.
- **Composable specialization.** Different adapters can encode different tools, personas, or domain expertise; the router combines them on demand. Directly aligned with agent stacks where each "capability" is a first-class object.
- **Cheaper than full MoE.** All the routing benefits at a fraction of the parameter cost — adapters share the base's FFNs and attention weights instead of duplicating them.

## Gotchas & tricks

- **Router capacity is the bottleneck.** A tiny linear router is fine at small `N`; past a few dozen adapters it becomes the failure point. Hierarchical routers (route to a family first, then to an adapter within the family) push the ceiling higher.
- **Adapter interference is real.** Two adapters attempting to modify the same subspace additively can cancel or amplify unpredictably. Orthogonality regularization between adapters is one lever; strict top-1 routing is another.
- **New-adapter cold start.** A freshly minted adapter has zero router mass at initialization — it must be forced-routed for a warmup period, then handed back to the learned router.
- **Storage grows linearly with skills.** Not free, just cheap. At `r = 16`, `N = 100` adapters over a 70B base is on the order of a few GB — much less than 100 full checkpoints, but not zero.
- **Doesn't fix task ordering effects.** The order in which you add adapters still matters if the router is fine-tuned each round — freeze the router periodically or use replay to stabilize.

## Sources

- Paper: *Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA* — Mind Lab, 2026 — the reference open realization at agent scale.
- Paper: *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., 2021 — the underlying adapter primitive.
- Related: DeepSeek-V3's aux-loss-free balancing — [../../architectures/aux-loss-free-balancing.md](../../architectures/aux-loss-free-balancing.md) — the routing-balance technique transfers directly.
