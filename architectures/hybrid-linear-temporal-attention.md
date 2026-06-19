# Hybrid Linear Temporal Attention

*Depth — Kairos's per-layer factorization of attention over time: sliding window + dilated sliding window + gated linear attention, with a provable bound on long-horizon error accumulation.*

**TL;DR:** World models that simulate long temporal trajectories run into two competing pressures: short-range temporal detail (frame-to-frame dynamics) needs dense local attention, but global world state (what's been happening for the last 10 minutes) needs persistent memory that doesn't explode in compute. Kairos resolves this with three attention paths *per layer* — **sliding window** for local dynamics, **dilated sliding window** for mid-range dependencies, and **gated linear attention** for persistent global memory. The Kairos paper provides a formal theoretical bound showing this factorization strictly limits error accumulation across extended horizons.

**Prereqs:** [attention](../fundamentals/attention.md), [mla](mla.md)
**Related:** [_normalization](_normalization.md)

---

## What it is

A specialization of hybrid-attention architectures (mixed dense + linear-attention designs like RWKV / Mamba-MoE hybrids) for the **temporal** axis specifically. Each layer reads the time axis through three complementary attention paths with different effective receptive fields, combined into a single per-layer output.

## How it works

For a sequence of $T$ frames (or frame embeddings), each layer computes three attention paths in parallel:

**1. Sliding-window attention** over the last $w$ steps. Standard local causal attention with window size $w$ (e.g. $w = 32$). $O(Tw)$ compute, $O(w)$ memory. Captures frame-to-frame dynamics and short-range coherence.

**2. Dilated sliding-window attention** over a window of $w$ steps with dilation factor $d$. Reads steps $\{t, t-d, t-2d, \ldots, t - (w-1)d\}$. Same compute as path 1 but with $d \times$ larger effective receptive field. Captures mid-range dependencies (a few seconds of behavior, a small action chunk).

**3. Gated linear attention** with a recurrent hidden state. Standard linear-attention recurrence with a learned gate on the state update:

$$S_t = g_t \odot S_{t-1} + K_t V_t^\top, \quad O_t = Q_t S_t$$

The gate $g_t$ controls how long information persists in $S$. Captures persistent global memory — the equivalent of "world state for the last 10 minutes" — at $O(1)$ per-step cost.

Outputs from the three paths are concatenated (or summed with learned weights) before the feed-forward block.

The error-accumulation bound (formal proof in the paper, not reproduced here) leans on the factorization: if any one path's error stays bounded over $T$ steps, and the three paths' errors don't constructively interfere, the combined model's rollout error grows at most linearly in $T$ rather than exponentially.

## Why it matters

- **Long-horizon rollouts without quadratic blowup.** Full dense attention over a 10-minute video at 30 FPS is 18,000 frames — quadratic attention is infeasible. The three-path factorization keeps the per-step cost constant.
- **Formal error bound.** Most video-generation world models have no theoretical guarantee about extended-rollout drift. The factorization gives one: error grows linearly, not exponentially, in $T$.
- **Aligns with real-world hardware constraints.** Each path is efficient on standard GPU kernels; the gated linear attention path means inference does not need to materialize the full history KV cache.
- **One layer, all three time scales.** Earlier hybrid designs (e.g. Jamba) interleave attention types across layers. Per-layer fusion gives every layer access to all three scales.

## Gotchas & tricks

- **Gate initialization matters.** A poorly-initialized gate either decays state too fast (no memory) or never decays (state explodes). The paper's recipe is not fully visible from the abstract; standard linear-attention initialization tricks (forget-bias init) likely apply.
- **The dilated window's dilation factor $d$ is a tuning knob.** Larger $d$ = bigger receptive field but coarser mid-range coverage. The paper's choice isn't disclosed in the abstract.
- **The three paths can be combined differently** (concat + linear, weighted sum, gated mixture). The right choice probably depends on the downstream task.
- **The error bound is for the factorization, not the data.** Real long-horizon rollouts can still drift due to compounding sampling noise; the bound says compounding is at worst linear, not that it's zero.

## Sources

- Paper: *Kairos: A Native World Model Stack for Physical AI* — Kairos Team, 2026 — [arXiv:2606.16533](https://arxiv.org/abs/2606.16533).
- Related: *Linear Attention* — Katharopoulos et al., 2020 — the linear-attention recurrence path 3 builds on.
- Related: *Longformer / BigBird* — Beltagy et al., Zaheer et al. — sliding + dilated attention as a sparse-attention motif.
