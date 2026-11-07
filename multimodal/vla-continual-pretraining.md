# VLA Continual Pretraining with 3D Representation Alignment (SLA)
*Depth — install 3D spatial reasoning into a trained VLA policy via continual pretraining, without re-training from scratch.*

**TL;DR:** A pretrained VLA policy (π0.5) is upgraded with 3D awareness by *continuing pretraining* with a joint loss: (a) the standard action-prediction loss, plus (b) a cosine-alignment loss between selected VLM visual tokens and 3D spatial embeddings from a frozen perception backbone. No change to the base architecture; action post-training follows. Reported SOTA on RoboDojo-Real, RoboTwin 2.0, RoboCasa-Panda, and LIBERO-Plus; +24.0-point average success-rate improvement over π0.5 across 8 real-world robot-arm conditions.

**Prereqs:** [vla](./vla.md)
**Related:** [vla-flash-decoding](./vla-flash-decoding.md)

---

## What it is

The typical VLA pretraining stack (vision encoder + LLM backbone + action head) learns good *2D* visual grounding by default, because the pretraining data is 2D images + text. But manipulation relies on 3D structure — depth, contact, object pose — which a 2D encoder represents only indirectly.

SLA installs this 3D structure *after* VLA pretraining, as a continual-pretraining stage. A frozen 3D perception model (depth + object-pose predictor) emits a set of spatial embeddings per scene; the VLA is asked, during continued pretraining, to make its own intermediate visual tokens cosine-align to those embeddings, while also still predicting actions.

## How it works

### Setup

- Base: pretrained π0.5 VLA policy (image encoder + LLM backbone + continuous action head).
- Teacher: a frozen 3D perception backbone (e.g. a depth + pose predictor). Produces spatial embeddings $z_{\text{3D}}$ per scene patch.
- Dataset: large-scale manipulation demonstration data that includes RGB and action labels; the 3D teacher is run offline to generate $z_{\text{3D}}$.

### Continual-pretraining loss

$$
\mathcal L = \underbrace{\mathcal L_{\text{action}}}_{\text{VLA action-prediction}} + \lambda \cdot \underbrace{(1 - \cos(z_{\text{VLM}}, z_{\text{3D}}))}_{\text{cosine alignment}}
$$

where $z_{\text{VLM}}$ is a selected set of intermediate visual tokens inside the LLM backbone. $\lambda$ balances alignment vs action quality; reported in the single-digit range.

The alignment is one-directional: the VLM moves toward the teacher's spatial embedding space, not vice versa. The teacher is frozen.

### Followed by action post-training

After the continual-pretraining stage, the policy is fine-tuned on action data with a standard VLA loss (no alignment term). The spatial alignment persists because the backbone's representations have been shaped.

## Why it matters

- **Post-hoc modality upgrade.** VLAs are expensive to pretrain. Installing a new modality (3D) without re-pretraining is an underexplored lever; SLA shows alignment with a frozen teacher is sufficient.
- **Clean separation.** Continual pretraining upgrades perception; action post-training upgrades control. Each stage has a clean loss and a clean data pipeline.
- **Transferable beyond robotics.** The pattern — align VLM tokens with a frozen specialist encoder during continual pretraining — plausibly transfers to any VLM where a strong unimodal specialist exists (audio, depth, segmentation, graphs).

## Gotchas & tricks

- **Alignment layer selection.** Cosine-aligning the *final* VLM visual tokens is too late (they are mixed with language). Aligning the *earliest* tokens is too early (they are low-level). Middle layers reported.
- **Teacher quality caps performance.** If the 3D teacher is noisy, the VLM inherits that noise. A good teacher is a prerequisite, not an afterthought.
- **Continual pretraining can forget.** The action-prediction term in the loss keeps the policy from collapsing, but underweighting it can degrade the base VLA's performance. Monitor action success on a held-out set throughout.
- **Not a fix for a bad VLA base.** SLA lifts an already-decent policy; starting from a weak VLA gives weak alignment gains.
- **Cosine, not L2.** Scale-free alignment (cosine) avoids forcing the VLM to match the teacher's magnitude; the direction is what carries the 3D structure.

## Sources

- Paper: *SLA: Scaling Spatial-Language-Action Model via Continual Pre-training* — Song et al., HKUST-GZ / BAAI / Tsinghua / OLA Dimensions, 2026 — [arXiv:2611.08373](https://arxiv.org/abs/2611.08373).
- Background: *π0.5* — Physical Intelligence — the base VLA.
- Background: [vla](./vla.md) — the model class SLA belongs to.
