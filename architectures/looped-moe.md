# Looped MoE (LOOM)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Looped transformers reuse the same block N times to add effective depth without extra parameters, but naive loops break after ~2 iterations: hidden-state variance grows and expert routing collapses onto a few experts. LOOM makes looped MoE scale to 9-12 loops with three fixes — scaled residual updates, input re-injection, and per-loop routers. A 700M-param LOOM drops perplexity 18.36 → 16.54 under near-iso-FLOP. Introduced in *Looping Beyond Twice* (2026, arXiv:2610.01153).

**Prereqs:** [_moe.md](_moe.md), [transformer-block.md](transformer-block.md)
**Related:** [deepseek-moe.md](deepseek-moe.md), [aux-loss-free-balancing.md](aux-loss-free-balancing.md), [capacity-factor.md](capacity-factor.md)

---

## What it is

A recipe for stacking many loops of a shared transformer block *where the block contains a MoE FFN*. Looped transformers for dense models had been studied as a parameter-efficient depth lever; LOOM identifies that MoE loops have *distinct* failure modes (expert selection collapse across loops) that dense loops don't, and shows that fixing them unlocks recurrent depth as a scaling axis orthogonal to parameters.

## How it works

Three fixes applied together:

1. **Scaled residual updates.** The residual connection's increment is divided by the loop count $L$:
   $$
   h^{(l+1)} = h^{(l)} + \frac{1}{L} \cdot \mathrm{Block}(h^{(l)})
   $$
   This keeps $\|h^{(l)}\|$ bounded as $L$ grows, avoiding the hidden-state variance explosion that otherwise kills training beyond ~2 loops.
2. **Input re-injection.** After each loop iteration, the layer's input $h^{(0)}$ is added back:
   $$
   h^{(l+1)} = \mathrm{ScaledResidual}(h^{(l)}) + h^{(0)}
   $$
   This anchors later iterations to the token embedding so the loop doesn't drift into a representation unrelated to the input.
3. **Per-loop routers.** Each loop iteration $l$ has its own router (and sometimes its own expert pool). This prevents the "selection collapse" where successive loops pick the same top-$K$ experts because they see similar hidden states.

Training uses standard MoE auxiliary-loss load balancing per router, inherited unchanged from DeepSeek-style fine-grained MoE.

## Why it matters

If recurrent depth is a free scaling axis, it reframes what "parameter-efficient scaling" means — especially for inference, where loops are serial but share weights. For memory-constrained deployment, LOOM lets a 700M-param checkpoint deliver effective capacity closer to a dense ~1.4B model.

Reported: stable scaling from 2 to 9-12 loops across 100M-1.7B dense-equivalent models. 700M-param LOOM: perplexity 18.36 → 16.54 under near-iso-FLOP, beating both deeper non-looped and shallower-but-wider MoE baselines at the same FLOP budget.

## Gotchas & tricks

- **The three fixes are coupled.** Dropping any one of them breaks training beyond a few loops; the paper's ablation shows no single fix suffices.
- **Per-loop routers add bookkeeping.** Memory for router parameters scales with $L$, though it's small relative to expert FFN weights.
- **Inference latency is serial in $L$.** Looped MoE trades parameter count for serial depth — great when memory-bound, less so when latency-bound.
- **Not a universal replacement for depth.** Beyond ~12 loops, returns taper and the two scaling axes (parameters vs loops) diverge in loss curves.
- **Doesn't change the standard load-balance machinery.** Each per-loop router runs its own auxiliary loss; the shared-expert concept from DeepSeekMoE transfers unchanged.

## Sources

- Paper: *Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts* — He, Li, Chang et al., 2026 — [arXiv:2610.01153](https://arxiv.org/abs/2610.01153).
