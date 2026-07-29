# Positive-Direction Matching (PDM)
*Depth — a branch-aware on-policy diffusion distillation objective that fixes the "Negative Branch Asymmetry" failure of naive CFG-composed matching.*

**TL;DR:** In on-policy diffusion distillation under classifier-free guidance, matching the *composed guided velocity* is under-identified at the branch level — positive- and negative-branch errors can compensate. PDM separates the objective into two terms: match the **positive prediction** directly, and match the **CFG conditional direction** (positive minus negative) rather than the sum. This eliminates the "positive-down, negative-up" pathology (**Negative Branch Asymmetry, NBA**) that appears when the teacher's negative branch carries privileged information the student can't reach.

**Prereqs:** [classifier-free-guidance.md](classifier-free-guidance.md), [on-policy-diffusion-distillation.md](on-policy-diffusion-distillation.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Naive OPD extends velocity matching to the CFG-composed prediction:

```
L_naive = || v_neg_S + w(v_pos_S − v_neg_S) − v_neg_T − w(v_pos_T − v_neg_T) ||²
```

That single scalar can go to zero even when `v_pos_S ≠ v_pos_T` and `v_neg_S ≠ v_neg_T`, provided the two errors *compensate*. Under a symmetric setting (teacher and student share the same negative conditioning) both branches decrease jointly and the compensation is benign. Under an asymmetric setting (the teacher's negative branch has privileged information — a stronger encoder, extra tokens, a modality the student can't see) the negative-branch error *grows* while the positive error decreases: NBA.

## How it works

PDM replaces the single composed loss with two branch-aware terms:

- **Positive prediction match:** `L_pos = || v_pos_S − v_pos_T ||²` — direct, unguided.
- **CFG direction match:** `L_dir = || (v_pos_S − v_neg_S) − (v_pos_T − v_neg_T) ||²` — matches the *guidance direction*, not the sum.

The total is `L = L_pos + λ · L_dir`. Because the positive term is unguided, the branch-error compensation of the naive loss is broken; because the direction term constrains the difference, the student cannot silently drift its negative branch to compensate the positive.

## Why it matters

On dense-to-sparse video control, naive guided matching is highly sensitive to inference guidance scales — the student either under-follows conditioning at low `w` or degrades at high `w`. PDM is robust across scales and transfers knowledge from teacher to student more effectively. More broadly, it names a previously-unlabeled instability in the dominant diffusion-distillation recipe, and the fix is a one-line change to the loss.

## Gotchas & tricks

- NBA only shows up when the teacher's negative branch is *asymmetric* — check the negative-conditioning setup before assuming naive OPD is safe.
- `L_pos` alone (drop `L_dir`) works but under-follows guidance at inference; both terms are needed.
- Choice of `λ` interacts with the teacher's guidance scale — start at `λ = 1` and sweep.
- PDM is orthogonal to warm-up, EMA, and consistency-style regularizers — combine freely.

## Sources

- Paper: *Rethinking Classifier-Free Guidance in On-Policy Diffusion Distillation* — Li et al., 2026 — [arXiv:2607.24731](https://arxiv.org/abs/2607.24731)
