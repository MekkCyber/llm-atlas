# Full-participation MoE (block-conditioned expert composition)
*Depth — decoupling participation from execution and materialization in MoE routing.*

**TL;DR:** Standard MoE couples three quantities: **participation** (how many experts contribute knowledge to a token's output), **execution** (how many experts are actually computed — the FLOP cost), and **materialization** (how many expert-sized parameter sets are stored — the memory cost). Top-K routing ties all three together: to have K experts participate, you must execute K and materialize N. IntBMoE (Cheng et al., 2026) decouples them by conditioning on a *block-level* signal and composing per-token effective experts from a fixed pool, so every materialized expert participates in every token's output while only a sparse subset is actually executed. A Dual-Path Residual Gating keeps the composed output stable when only a subset fires.

**Prereqs:** [_moe.md](./_moe.md), [deepseek-moe.md](./deepseek-moe.md)
**Related:** [aux-loss-free-balancing.md](./aux-loss-free-balancing.md) · [capacity-factor.md](./capacity-factor.md) · [load-balancing-loss.md](./load-balancing-loss.md)

---

## What it is

A design point that breaks the standard MoE constraint *participation ≤ execution ≤ materialization ≤ N*. In IntBMoE, participation can equal materialization (every expert contributes) while execution stays sparse (only a subset is computed).

## How it works

- **Block-level conditioning.** A per-block conditioning vector (derived from the token's hidden state) is used to compose the effective expert for that token as a weighted combination of expert-blocks. This composition is dense at the *contribution* level.
- **Sparse execution.** Only expert-blocks whose composition weight exceeds a threshold are actually computed; the rest are represented by their static bias / residual contribution. Execution stays top-K-like in FLOPs.
- **Dual-Path Residual Gating.** The layer combines an executed path (sparse compute over selected blocks) with a residual bias path (constant per block, materialized). This second path is what keeps the *contribution* of non-fired experts nonzero without paying their compute — full participation, sparse execution.

The three axes at a glance:

| Axis | Standard top-K MoE | IntBMoE |
| --- | --- | --- |
| Participation | K (only fired experts contribute) | N (all materialized experts contribute) |
| Execution | K (only fired experts computed) | K (unchanged) |
| Materialization | N (stored) | N (stored) |

## Why it matters

- **Better quality at same FLOPs.** Empirically, having more experts *contribute* per token yields measurable gains without changing execution cost. In production A/B testing on AMap recommendation, IntBMoE gave +2.4% relative UVCTR — a nontrivial lift for a live surface.
- **Cleaner conceptual axis.** The participation / execution / materialization split is a useful frame for reasoning about MoE variants and their tradeoffs — several existing tricks (aux-loss-free balancing, capacity factor) live on the execution axis; IntBMoE opens the participation axis.
- **Complementary to routing algorithm.** Nothing in IntBMoE forbids stacking with expert-choice routing, aux-loss-free balancing, or fine-grained expert granularity — the block-conditioning layer is orthogonal to router design.

## Gotchas & tricks

- **Memory cost of full materialization.** Because every expert must be stored (participation = materialization = N), memory scales with the total expert count, same as standard MoE. This is not a memory saver.
- **Non-fired expert contributions must be bounded.** If the residual bias path is too large it swamps the executed path; too small and full participation is nominal. The gating design is where the wins actually come from — casual "add a residual" doesn't reproduce it.
- **Serving matches training.** The composed-then-sparse-execute path introduces a routing-and-composition step inside every MoE layer. Deployment must implement this exact pipeline or the pretrained composition weights are meaningless.
- **Distribution-shift sensitivity.** Because non-executed experts still contribute via a fixed residual, the layer is more sensitive to activation distribution shift than pure top-K — the residual assumes a training-time expected contribution. Careful calibration or fine-tuning after distribution shift is prudent.

## Sources

- Paper: *IntBMoE: Integrating Block-Level Conditioning into Expert Composition for Full-Participation Mixture-of-Experts* — Cheng et al., AMap / Alibaba, 2026 — [arXiv:2609.21346](https://arxiv.org/abs/2609.21346).
- Related: DeepSeekMoE (fine-grained + shared experts) — see [deepseek-moe.md](./deepseek-moe.md).
