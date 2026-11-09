# VLA spatial alignment via activation distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Vision-language-action (VLA) policies built on 2D VLM backbones struggle with 3D spatial reasoning but can't afford a 3D perception tower at every inference step. VLA spatial alignment adds a training-only auxiliary loss that distills features from a frozen 3D perception model into chosen intermediate layers of the policy; the alignment head is thrown away at deployment, so inference cost matches the base VLA. Introduced in SLA (2611.08373).

**Prereqs:** [mid-training](../pre-training/mid-training.md)
**Related:** [parallel-primitive-decoding](../inference/parallel-primitive-decoding.md)

---

## What it is

A **continual pre-training recipe** for VLAs with two components discarded at inference:
- An alignment loss between policy activations and 3D features from a frozen backbone (OmniVGGT in SLA).
- Adaptive layer selection — the subset of policy layers whose activations get aligned is learned, not fixed.

At deployment, the auxiliary tower, alignment head, and layer-selection gates are removed; the policy that remains has the same architecture and FLOPs as its base (e.g. π0.5).

## How it works

Given a policy $\pi_\theta$ initialized from π0.5 and a frozen 3D feature extractor $f_{3D}$:

1. For each training batch, compute 3D features $z = f_{3D}(\text{obs})$.
2. For each policy layer $\ell \in \mathcal{L}$, extract activation $a_\ell$. A learned soft gate $g_\ell$ selects which layers to align.
3. Alignment loss: $\mathcal{L}_{\text{align}} = \sum_\ell g_\ell \cdot d(\phi_\ell(a_\ell), z)$ where $\phi_\ell$ is a small projection (training-only) and $d$ is cosine or MSE.
4. Total loss: $\mathcal{L}_{\text{action}} + \lambda \mathcal{L}_{\text{align}}$. Also used: state dropout (randomly mask proprioception so the policy can't lean on it) and training on a heterogeneous union of robot datasets with embodiment conditioning.

The gate $g_\ell$ learns to concentrate alignment pressure on layers where 3D signal is most useful — typically mid-depth layers, per the paper's reported ablations.

## Why it matters

- **No inference tax.** Spatial priors become implicit features of the policy, not an extra forward pass.
- **Generalizes beyond 3D.** Any perception prior exposed as frozen features — depth, flow, semantic segmentation — can be slotted in.
- **Reusable with any policy backbone.** SLA uses π0.5, but the recipe is base-agnostic.

Reported gains: SOTA on RoboDojo-Real, RoboTwin 2.0 Clean2Random, RoboCasa-Panda, LIBERO-Plus; +24.0 points over π0.5 on real-robot evaluation across 8 conditions.

## Gotchas & tricks

- The alignment loss can dominate if $\lambda$ is too high; keep it as a soft anchor, not a hard constraint.
- **State dropout** matters: without it, the policy memorizes proprioception shortcuts and the aligned vision features never become load-bearing.
- Heterogeneous robot data needs **embodiment-aware conditioning** (prompt text or embedding) so the alignment signal doesn't fight morphology-specific action distributions.

## Sources

- Paper: SLA: Scaling Spatial-Language-Action Model via Continual Pre-training — Song et al., 2026 — [arXiv:2611.08373](https://arxiv.org/abs/2611.08373)
