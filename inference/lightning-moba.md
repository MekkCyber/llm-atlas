# LightningMoBA
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An inference-time acceleration of **Mixture of Block Attention (MoBA)** with three changes: (i) dynamic routing via a score *threshold* instead of fixed top-$k$, (ii) a **KV-centric** execution plan that groups work by (query head, KV block) and bounds per-task fanout, and (iii) runtime pipelining that overlaps K/V transfers with Tensor Core compute on Hopper GPUs. Delivers **1.20–1.60× end-to-end generation speedup** at matched Top-16 retention across five checkpoints, with mean RULER rising 74.42 → 75.28.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [../architectures/_sparse-attention.md](../architectures/_sparse-attention.md), [../architectures/moba.md](../architectures/moba.md)
**Related:** [_speculative-decoding.md](_speculative-decoding.md)

---

## What it is

A drop-in replacement for the MoBA attention kernel on Hopper-class GPUs. MoBA splits KV into blocks and routes each query to top-$k$ blocks; LightningMoBA keeps the model unchanged but reshapes the *scheduling* so block-selection becomes adaptive and the kernel runs closer to peak.

## How it works

1. **Dynamic routing (threshold).** Replace "pick the top-$k$ blocks per query" with "pick every block whose score exceeds $\tau$" (plus a floor $k_{\min}$ for pathological queries). Easy queries use fewer blocks; hard queries use more. $\tau$ is calibrated per head on a small calibration set and held fixed at inference.
2. **KV-centric execution.** Standard MoBA kernels iterate queries and gather KV blocks per-query, which is low-locality: adjacent queries read overlapping KV. LightningMoBA pivots: it groups work by (query head, KV block) columns, so each KV block is streamed once and consumed by many queries. High-fanout columns (one block, many consumers) are split into bounded tasks to avoid stragglers.
3. **Runtime pipelining.** On Hopper, K/V transfers use the TMA (Tensor Memory Accelerator) while the WGMMA Tensor Cores compute. LightningMoBA overlaps these so K/V arrives just-in-time for each tile's matmul, hiding transfer latency behind compute.

## Why it matters

- **MoBA (Kimi, 2025) is the leading open-source sparse-attention mechanism** for long context. Cutting its inference cost 1.5× without quality loss makes it a stronger alternative to sliding-window + dense recall hybrids.
- **Quality *improves*.** Dynamic routing trims noisy blocks that fixed top-$k$ would have included on easy queries; RULER mean rises 74.42 → 75.28 at matched block budget.
- **Pure inference-time change.** No retraining — drop-in on existing MoBA-trained checkpoints (Kimi-K1.5, derivatives).

## Gotchas & tricks

- **Threshold calibration matters.** $\tau$ set from a mismatched calibration distribution can over- or under-recruit blocks. The paper reports $\tau$ per head, calibrated on in-distribution long-context prompts.
- **KV-centric pivot assumes enough fanout.** Short contexts have few KV blocks per query; the pivot's win shrinks. LightningMoBA's speedups are measured in long-context regimes.
- **Hopper-specific kernels.** The pipelining relies on TMA + WGMMA; benefits on Ampere / Blackwell depend on analogous primitives.
- **Not evaluated with non-MoBA checkpoints.** The "quality rises" finding depends on MoBA's training distribution — don't assume retrofitting a vanilla sparse-attention model yields the same effect.

## Sources

- Paper: *Accelerating Mixture of Block Attention via Dynamic Routing, KV-Centric Execution and Runtime Pipelining* — Tencent (GitHub org), 2026 — primary source.
- Paper: *MoBA* — Kimi, 2025 — the attention mechanism this paper accelerates.
