# Hybrid Linear-Attention Distillation
*Depth — distilling a pretrained Transformer into a hybrid linear-attention student with principled initialization.*

**TL;DR:** Converting a pretrained Transformer to a model that mixes softmax attention with **linear / recurrent attention** (e.g. Gated DeltaNet, Mamba) traditionally requires long distillation runs because the new recurrent parameters start randomly and have to be repaired during training. **Taylor-Calibrate** (Zhou et al., 2026) initializes those recurrent parameters from a **Taylor expansion of the teacher's attention statistics** and adds a brief **layer-local alignment pass** before global distillation — cutting distillation tokens by **4.9× to 9.2×**.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [transformer-block](transformer-block.md)
**Related:** *(GDN / linear-attention depth files as they land)*

---

## What it is

Hybrid linear-attention models replace some or all softmax-attention layers with a linear / recurrent variant — **Gated DeltaNet (GDN)**, Mamba, GLA, RWKV — that has constant-state inference cost (no KV cache growth). The standard conversion recipe:

1. Take a pretrained Transformer teacher.
2. Replace selected attention layers with linear-attention student layers (same hidden size).
3. **Copy QKV projections** from the teacher; initialize student-specific parameters (memory decay, write gate, output gate for GDN) randomly or with heuristics.
4. Distill end-to-end (student matches teacher logits / hidden states) on a corpus.

The problem: step 3's random init leaves the student's recurrent parameters far from anything teacher-compatible, so the early distillation steps just **patch up the bad init** instead of learning useful representations. Token cost balloons.

Taylor-Calibrate replaces step 3 with a **principled init** derived from teacher behavior, then adds a **layer-local alignment** before step 4.

## How it works

### Stage 1 — Taylor-guided initialization

Teacher softmax attention can be written as $\text{Attn}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d}) V$. A first-order Taylor expansion (around the row-mean of $QK^\top / \sqrt{d}$) approximates this as a **linear function of $V$**:

$$
\text{Attn}(Q, K, V) \approx M(Q, K) \cdot V
$$

for some matrix $M(Q, K)$ derived from the local statistics of $QK^\top$. Crucially, $M$ can be parameterized as a **gated linear recurrence** of the form GDN uses, and its parameters (memory decay $\alpha$, write gate $w$, output gate $o$) can be read off directly from teacher attention statistics:

```
α ← row-wise effective context length of teacher attention
w ← row-wise sum of softmax weights (≈ 1 in standard attention)
o ← projection that matches teacher's downstream effective output magnitude
```

The exact derivation matches per-layer teacher statistics gathered from a small forward pass over the calibration set.

### Stage 2 — Layer-local alignment

Before global distillation, do a brief **per-layer** alignment pass:

For each replaced layer $\ell$:

```
freeze all other layers
for batch in calibration_set:
    h_teacher = teacher_layer_ℓ(input)
    h_student = student_layer_ℓ(input)
    minimize ||h_student - h_teacher||
```

A few hundred steps per layer is enough to close most of the residual gap from the Taylor init. The forward passes can be shared across layers, so total wall-clock is modest.

### Stage 3 — Standard end-to-end distillation

With the student starting at a near-teacher state, the standard end-to-end distillation (KL on logits + optional hidden-state matching) converges in **4.9× to 9.2×** fewer tokens to reach target quality.

## Why it matters

- Distillation cost is the gating factor for hybrid-linear-attention adoption. A 5–9× reduction makes "convert your existing Transformer to a hybrid" routine.
- Demonstrates a **generalizable pattern**: teacher → student-with-different-parameterization distillation should start from a closed-form derivation of the student parameters that approximate the teacher, not from random init.
- Validates **Gated DeltaNet** as a competitive linear-attention substitute when properly initialized — most prior failures were init failures, not architectural ones.

## Gotchas & tricks

- **Teacher statistics must come from real data.** Calibration set has to cover the deployment distribution; init derived from out-of-distribution attention statistics regresses.
- **Layer-local alignment can plateau.** Some layers (esp. the last 2–3) don't fully align via local matching because they encode long-range dependencies that GDN approximates differently. The end-to-end stage 3 mops these up.
- **Doesn't work for arbitrary student parameterizations.** The Taylor expansion has to map cleanly to the student's parameter form. GDN works because it has the right gated-recurrence shape; arbitrary attention variants need their own derivation.
- **Calibration size matters.** Too small (< 1M tokens) and the teacher-statistics init is noisy; too large adds wall-clock without benefit. Paper reports a sweet spot.
- **Hybrid ratio matters.** Keeping a few full-attention layers (typically every $K$-th) preserves long-range capability; full conversion underperforms even with Taylor-Calibrate.

## Sources

- Paper: *Taylor-Calibrate: Principled Initialization for Hybrid Linear Attention Distillation* — Zhou, Wu, Wang, Mishra, Song, Athiwaratkun, Xu (Sydney, Together AI, Berkeley, UT Austin, Microsoft), 2026, arXiv 2606.16429.
- Paper: *Gated DeltaNet* — primary linear-attention variant Taylor-Calibrate targets as student.
- Paper: *Mamba* / SSM family — adjacent linear-attention substitutes Taylor-Calibrate could in principle apply to.
- Background: standard Transformer → SSM distillation literature (MambaInLlama, etc.).
