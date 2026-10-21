# Kronecker Delta Rule
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A low-rank update rule for the matrix-valued memory in linear attention. The standard delta rule rewrites the full residual between the stored and target value along the direction of the incoming key; the Kronecker delta rule rewrites only a *single scalar association* along the outer product $k \otimes v$. Same bounded-norm stability, far less interference with previously-stored key→value associations. Builds **KronLSTM** on top; at 1.5B / 100B tokens it beats linear-attention and SSM baselines on recall-heavy long-context tasks.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [_linear-attention](_linear-attention.md)

---

## What it is

Linear attention stores sequence state as a $d_k \times d_v$ matrix memory $S_t$ and reads it by $v_t = S_t^\top q_t$. The *delta rule* updates $S$ to correct the residual between the current prediction and the target value for an incoming key. In the DeltaNet formulation, $S_t = S_{t-1} + \beta_t (v_t - S_{t-1}^\top k_t) k_t^\top$: every entry correlated with $k_t$ is rewritten.

The Kronecker delta rule reformulates this as a *rank-one Kronecker outer product*: write only $S_t = S_{t-1} + \beta_t (v_t - S_{t-1}^\top k_t)^\star k_t^\top$ where the correction is projected onto the explicit $k \otimes v$ direction, i.e. a single scalar association. Entries orthogonal to $k_t \otimes v_t$ are left intact.

## How it works

The paper proves two anchor results for an associative matrix memory updated by this rule:

- **Convergence to a steady state.** The expected $S_t$ converges exponentially to a fixed matrix determined by the (stationary) key/value distribution, under a mild non-degeneracy condition on $k_t$.
- **Bounded expected memory norm.** $\mathbb{E}\|S_t\|_F$ stays bounded independent of $t$, matching the delta-rule guarantee.

**KronLSTM** adds three gates on top of the raw rule:
- A **vector forget gate** $f_t \odot S_{t-1}$ (per-row scalars, cheaper than a full matrix forget).
- An **erase gate** that zeros out specific associations before writing (lets the model unlearn one slot without touching others).
- A **write-target gate** that controls how much the Kronecker update is applied this step.

The outer loop is still a linear-attention recurrence — $O(d^2)$ state per layer, $O(1)$ per-token compute independent of sequence length. KronLSTM slots into the "chunkwise parallel scan" machinery already used for Mamba and DeltaNet.

## Why it matters

Linear-attention / SSM variants keep hitting a *recall ceiling* on needle-in-a-haystack and real-world long-context tasks. The paper attributes a chunk of this to update-rule interference: the standard delta rule's full-vector overwrite destroys unrelated associations stored earlier. The Kronecker rule is a cheap, local, theory-backed fix.

Experimentally: at **1.5B parameters / 100B tokens**, KronLSTM beats Mamba, DeltaNet, and Gated-DeltaNet on language modelling perplexity, real-world recall benchmarks, and needle-in-a-haystack retrieval. The gap *widens* with longer context — exactly the regime where delta-rule interference bites hardest.

## Gotchas & tricks

- **Scalar correction isn't optional.** The whole win comes from projecting the correction onto $k \otimes v$. Correcting on the full residual direction recovers vanilla DeltaNet behaviour, interference and all.
- **Vector (not matrix) forget gate.** The paper uses per-row scalar forgets. A full matrix forget wipes previously-stored associations, cancelling the Kronecker gain.
- **Steady-state convergence needs non-degenerate keys.** If keys collapse to a low-dimensional manifold during training, the steady state degrades. QK-norm or similar key-normalisation helps.
- **Chunkwise parallel scan still applies.** Training throughput matches DeltaNet / Mamba; the rule-change doesn't break hardware-efficient implementations.
- **Hybridisation with full attention helps.** Like Mamba-2 and friends, interleaving KronLSTM blocks with occasional softmax-attention blocks closes the remaining recall gap at the long tail.

## Sources

- Paper: *Kronecker Delta Rule for Long Context Linear Attention* — Zólyomi, List, Klambauer, Böck, Hochreiter, 2026 — [arXiv:2610.26834](https://arxiv.org/abs/2610.26834).
- Background: *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* — Yang, Wang, Shen, Panda, Kim, 2024 — the DeltaNet lineage this rule extends.
