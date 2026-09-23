# Complex KDA (CKDA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A one-line extension of Kimi Delta Attention (KDA): widen the channel-wise gate range from $[0, 1]$ to $[-1, 1]$ *and* widen the delta-rule coefficient $\beta$ from $[0, 1]$ to $[0, 2]$. With both extensions, a **single** KDA step can realize a full 2D rotation (via a delta transformation composed with the gate's reflection) — matching DeltaProduct₂'s expressivity at KDA's single-step cost, and provably covering *every* orthogonal diagonal-plus-rank-one transition matrix.

**Prereqs:** [kimi-delta-attention.md](kimi-delta-attention.md)
**Related:** [../case-studies/kimi-k1-5.md](../case-studies/kimi-k1-5.md) · [mla.md](mla.md)

---

## What it is

Prior work on delta-rule linear RNNs showed that reaching 2D-rotation expressivity requires composing **two** delta transformations per step (DeltaProduct₂), doubling the state-update cost. CKDA achieves the same expressivity in a **single** step by re-using an already-present component of KDA — the channel-wise gate — as the second orthogonal ingredient (a reflection).

The trick has two independent range extensions, both of which are needed:

- **Gate range** $g_t \in [-1, 1]$ (was $[0, 1]$): allows the gate to *flip* a state channel, i.e. act as a reflection along that axis.
- **Delta coefficient** $\beta_t \in [0, 2]$ (was $[0, 1]$): allows the delta-rule step to over- or under-shoot the target, extending the family of delta transformations reachable in one step.

Together, gate-reflection ∘ delta-rotation covers every orthogonal diagonal-plus-rank-one matrix.

## How it works

**Reflection from a signed gate.** A gate value $g_i = -1$ on channel $i$ flips $S_{t-1}[i, :]$ — a reflection on that axis. On the rest of the channels the sign structure of $g$ picks out a subspace to reflect.

**Rotation from the delta rule.** With $\beta \in [0, 2]$, the delta step can realize the delta-rule half of an orthogonal decomposition.

**Composition.** Gate-first-then-delta ≡ reflection-then-rotation = rotation-with-a-signed-frame. The paper proves (theorem-style) that *every* orthogonal $D+ku^\top$ transition matrix is exactly a CKDA transition for some choice of $(g, \beta, k)$.

Cost stays at a single KDA update ($O(d)$ or $O(d^2)$ per step depending on parameterization), vs. DeltaProduct₂'s $2\times$. Stability is preserved: the extended ranges keep transitions non-expansive.

## Why it matters

- **Free expressivity.** Doubles the state-tracking reach without adding compute, memory, or a new operator — just a change in a parameter clamp.
- **State tracking on the state of the art.** A single CKDA layer tracks every finite group isomorphic to a subgroup of $SO(3)$ — a class of algorithmic tasks that trips up standard Transformers.
- **Depth savings.** Many state-tracking results the paper reports use one fewer layer for CKDA than for other diagonal-plus-rank-one linear RNNs at matched task difficulty.
- **Language-modeling holds up.** In LM benchmarks, CKDA beats Transformers and other linear RNNs, matches the KDA baseline, and shows promising scaling — the expressivity win doesn't cost quality.

## Gotchas & tricks

- **Both extensions are required.** Just widening $g$ or just widening $\beta$ does not give the full orthogonal class — you need the reflection *and* the rotation to compose.
- **Length extrapolation improves.** On S₃, S₄, and periodic audio continuation, the widened-range CKDA gives the strongest length extrapolation among the tested KDA settings — a practical training test.
- **Backprop stability under the new ranges.** With $\beta$ up to 2 and $g \in [-1, 1]$, gradients can pass through sign flips and near-doubling steps; a tanh-style parameterization keeps optimization well-behaved.
- **Not a Transformer replacement everywhere.** Same caveat as KDA — long-range exact retrieval still favors attention or a hybrid.
- **Retrofit-friendly.** Existing KDA-trained checkpoints can be *fine-tuned* into CKDA by simply widening the range clamps, no new parameters to initialize.

## Sources

- Paper: *Complex KDA: Understanding and Enhancing the Expressivity of Kimi Delta Attention* — Siems, Grazzi, Pöppel, Singh, Zela, Carstensen, Jitsev, Hutter, Cevher, Orvieto, Klein — ELLIS Institute Tübingen, 2026. [arXiv:2609.24797](https://arxiv.org/abs/2609.24797)
- Background: DeltaProduct / DeltaProduct₂ — the earlier two-step composition CKDA supersedes.
- Prereq: KDA (see [kimi-delta-attention.md](kimi-delta-attention.md)).
