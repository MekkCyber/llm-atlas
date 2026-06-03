# Sparse Attention for Long-Video Diffusion (LVSA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Training-free block-sparse attention for long-video diffusion transformers. Each query frame attends to (a) a small set of equispaced **global anchor frames** and (b) a local temporal **window**. The global set **rotates by one position per denoising step** so no frame is permanently demoted from anchor status — eliminating the fixed-grid bias that causes long-range temporal artifacts. Up to 3.3× wall-time speedup on Wan / HunyuanVideo and enables 257-frame generation that's OOM with dense attention on a single 80 GB GPU.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [../fundamentals/rope.md](../fundamentals/rope.md)
**Related:** [../architectures/mla.md](../architectures/mla.md), [_speculative-decoding](_speculative-decoding.md)

---

## What it is

Dense self-attention in video DiTs is $O(N^2)$ in the number of latent tokens $N = T \cdot P$ (frames × spatial patches). Past the training horizon (Wan: 81 frames; HunyuanVideo: 129), dense attention collapses to *near-static* output — the model loops or freezes. Earlier sparse-attention methods (Sparse VideoGen, AdaSpa, Sliding Tile, Radial) reduce compute but still leave long-range temporal artifacts because their patterns are *fixed*. LVSA fixes both compute and quality with a static pattern plus a rotation schedule across denoising steps.

---

## How it works

### Attention support per query frame

For query frame $t \in \{0, \ldots, T-1\}$, attend to $\mathcal{A}(t) = G^s \cup \mathcal{W}(t)$:

- **Global anchors $G^s$** at denoising step $s$:
  $$G^s = \{(s \bmod T_{per} + i \cdot T_{per}) \bmod T \mid i = 0, \ldots, \lceil T/T_{per}\rceil - 1\}$$
  Equispaced, shifted by one position per denoising step. Over $T_{per}$ consecutive steps every frame appears at least once as an anchor.
- **Local window $\mathcal{W}(t) = [t-W, t+W]$** of radius $W$, with adaptive bounds: when the window overlaps $G^s$, extend it on whichever side has room so the per-query budget stays constant at $|\mathcal{A}(t)| = |G| + \min(2W+1, T-|G|) \approx C$.

Default $C$ = the model's training-horizon frame count (Wan 2.1 1.3B: $C = (81-1)/4 + 1 = 21$); $T_{per}$ derived from $C$ and $W$.

Total complexity per denoising step: $O(T \cdot C \cdot P^2 \cdot d)$ — **linear in $T$**.

### Rotating global anchors

A fixed periodic $G$ would systematically impoverish non-anchor frames' representations over $S$ denoising steps — exactly the cause of long-range artifacts. Shifting $G^s$ by one position per step (modulo $T$) gives every frame anchor status at least once per cycle while keeping $|G^s|$ constant. Frame 0 stays a permanent anchor for scene-establishing content.

### Kernels

LVSA + the FlashInfer block-sparse kernel: 3.17× on Wan 2.1 1.3B at 6× horizon, 2.98× on Wan 2.1 14B at 6×, 3.33× on HunyuanVideo 1.5 at 1.5×. Per-step index recomputation is pure CPU and <1 ms — negligible vs the attention kernel.

---

## Why it matters

- **Compute *and* quality together.** Prior training-free extrapolation methods either reduced compute *or* preserved quality at extended horizons, not both. RIFLEx (RoPE-frequency tweak) doesn't reduce FLOPs; UltraViCo (logit decay) needs full dense attention. LVSA wins on both axes: 2.4× faster than dense at 4× horizon and +9.9 VQeval over dense.
- **Enables otherwise-infeasible generation.** HunyuanVideo 1.5 at 2× horizon (257 frames) OOMs dense on 80 GB; LVSA fits in ~60 GB.
- **Rotating-anchor pattern is the reusable idea.** Static sparse patterns have a fundamental fairness problem in iterative-refinement settings (diffusion) — some tokens always lose. Rotation makes a static pattern *temporally fair* without any per-step adaptation cost.
- **Cross-hardware.** Ports to NPU (2.71× on Wan 2.2 A14B) with the same recipe.

---

## Gotchas & tricks

- **Watch for static-rewarding eval bias.** VBench-Long's subject/background consistency dimensions reward frozen output — exactly dense attention's failure mode at extended horizons. LVSA's authors introduced VQeval to expose this; if you measure on VBench-Long alone you'll *look worse* while actually generating better video.
- **Expand the window when it overlaps anchors.** A naive sliding window wastes budget on global frames that are already attended to. The adaptive expansion keeps $|\mathcal{A}(t)|$ at the target $C$.
- **$T_{per}$ controls rotation period.** Too small ($T_{per} < S$ / many) and anchors revisit too often; too large and most denoising steps spend on the same anchors. Set $T_{per} = \lceil T/(C-(2W+1)) \rceil$ as a default.
- **Doesn't fix multi-scene generation.** The paper's analysis assumes single-scene rollouts; scene-change handling is open.
- **Orthogonal to MLA-style KV compression** ([VideoMLA](../architectures/mla.md)). Sparse attention reduces sequence length; MLA shrinks per-token KV. Compose them.

---

## Sources

- Paper: *LVSA: Training-Free Sparse Attention for Long Video Diffusion* — Glorian, Lamprou, Zhang, Yuan, Liu — Huawei Paris Research Center, 2026 — [arXiv:2605.31057](https://arxiv.org/abs/2605.31057).
- Code: https://github.com/JiusiServe/LongVideoSparseAttention
- Background: *RIFLEx* (Zhao 2025), *UltraViCo* (Zhao 2025), *Radial Attention* (Li 2025), *Sliding Tile Attention* (Zhang 2025) — predecessor training-free patterns.
- Models tested: Wan 2.1 1.3B/14B, Wan 2.2 A14B, HunyuanVideo 1.5.
