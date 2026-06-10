# Muon
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A matrix-aware optimizer for the 2D weight matrices of a Transformer. Maintains a Polyak-style momentum buffer like Adam, then **orthogonalizes the update matrix** before applying it — empirically a Newton–Schulz iteration that pushes the momentum buffer toward its nearest semi-orthogonal matrix. Used on the 2D weights only; embeddings and norms keep AdamW. About **~2× the training efficiency of AdamW** on LLM pretraining at matched validation loss, and the curvature-perspective analysis (Wang et al., 2026) explains why: matched first-order gain, but a much **smaller second-order curvature penalty per step**.

**Prereqs:** [attention.md](attention.md)
**Related:** [_optimizers.md](_optimizers.md) · [../pre-training/_training-stability.md](../pre-training/_training-stability.md) · [../pre-training/_lr-schedules.md](../pre-training/_lr-schedules.md)

---

## What it is

Adam/AdamW treats every parameter scalar independently — diagonal preconditioning by running second-moment estimates. For Transformer **weight matrices**, that throws away matrix structure: the rows and columns are coupled (they multiply activations together), and a diagonal preconditioner can produce updates with very anisotropic singular spectra. Anisotropic updates fight curvature.

Muon keeps a momentum buffer $M$ exactly like Adam, but applies a **whitening step**: compute an approximate orthogonalization of $M$ (via a few Newton–Schulz iterations) and use the orthogonalized matrix as the update direction. The result has nearly-uniform singular values, which makes the step length roughly the same in every direction the matrix can move.

Apply to 2D matrices (attention, MLP, output projections). Keep AdamW for the 1D parameters (embeddings, norms, biases) — orthogonalization is undefined / unhelpful there.

## How it works

For each matrix-shaped parameter $W$ at step $t$:

1. Compute gradient $G_t$.
2. Update momentum: $M_t = \mu M_{t-1} + G_t$.
3. **Orthogonalize:** $U_t = \mathrm{NewtonSchulz}(M_t, \text{iters}=5)$. Five iterations of a Newton–Schulz polynomial drive the singular values toward 1 fast (cheap in bf16 — no SVD).
4. Apply: $W_t = W_{t-1} - \eta \cdot U_t$.

Newton–Schulz on $M$:
$$
X_{k+1} = \tfrac{3}{2} X_k - \tfrac{1}{2} X_k X_k^\top X_k
$$
starting from $X_0 = M / \|M\|$ (spectral or Frobenius norm). Each iteration is two matmuls — cheap.

The curvature-perspective analysis decomposes one-step loss change as $-\eta \langle G, U\rangle + \tfrac{\eta^2}{2} U^\top H U + O(\eta^3)$. Both Adam and Muon get comparable first-order gain at matched validation loss; Muon's $U^\top H U$ term is smaller because the orthogonal step has near-uniform singular values and so spreads probe energy across Hessian eigendirections more evenly.

## Why it matters

- **~2× pretraining wall-clock at matched loss** vs AdamW on Transformer LLMs of the sizes where it's been tested. Gains compose with WSD schedules and µP-style scaling.
- **Curvature framing predicts where Muon wins:** landscapes with sharp dominant Hessian directions (most of LLM pretraining). Predicts where it won't help: ill-conditioned tasks where the curvature term isn't the bottleneck.
- **Drop-in for the 2D parameters.** No new tensors per parameter (no $v$-buffer like Adam), so memory cost is lower than Adam too — momentum-only.

## Gotchas & tricks

- **Only for 2D weights.** Embeddings, biases, RMSNorm scales: keep AdamW. Mixing is the default.
- **Newton–Schulz precision.** The fixed point assumes $\|X_0\| \le 1$; without the initial normalization the iteration diverges. Use spectral norm if you have it cheaply, else a Frobenius bound.
- **LR transfer differs from Adam.** The orthogonal update has a roughly constant operator norm, so the *effective* per-coordinate step is set by $\eta$ alone (no $v$ rescaling). LR tuned for AdamW typically does not transfer one-for-one — Muon LRs are often noticeably larger.
- **Layerwise weight decay** still applies; it's not part of the Newton–Schulz step and should stay on the parameter, not the update.
- **Sharding.** Newton–Schulz needs a global view of each matrix; under FSDP/TP you need to gather the parameter shards (or use a sharded variant) before orthogonalizing.

## Sources

- Paper: *Why Muon Outperforms Adam: A Curvature Perspective* — Wang, Zhang, Li, Bergemann, Yang — NUS / Yale / U. Minnesota, 2026 — arXiv 2606.04662 — second-order Taylor decomposition of the one-step loss change.
- Blog / reference impl: Keller Jordan's *Muon* — the Newton–Schulz orthogonalization and 2D-only application pattern.
