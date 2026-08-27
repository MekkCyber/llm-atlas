# Case Study: GigaBrain-0.7

*An embodied foundation model that unifies understanding, prediction, and action in a **three-system architecture** and scales pretraining to **37,000+ hours** of heterogeneous embodied data. Trained with **one-stage alignment** rather than the standard cascade of "pretrain VLM → bolt on action head". Substantial gains over its predecessor and π₀.₅ across zero-shot capability, instruction following, and post-training task success, on the in-house Maker H01 humanoid and mainstream embodiments.*

**Related concepts:** [three-system-vla](../multimodal/three-system-vla.md) · [_vla](../multimodal/_vla.md) · [../multimodal/README](../multimodal/README.md) · [qwen2-5 case study](qwen2-5.md)

---

## What this is

GigaBrain-0.7, released 2026 by the GigaBrain team (60+ authors). A vision-language-action (VLA) foundation model built as a single trainable system spanning three coupled subsystems:

- **System 1** — vision-language understanding (the VLM backbone).
- **System 2** — world prediction (forecasting the visual consequences of an action).
- **System 3** — action generation (multi-embodiment continuous control).

The paper's argument: prior VLAs have treated action generation as a bolt-on head after a fixed VLM. GigaBrain-0.7 trains all three systems together via **one-stage alignment**, letting understanding, prediction, and control co-adapt rather than compose. Pretraining spans 37,000+ hours of heterogeneous embodied data (human egocentric video, robot trajectories from many embodiments, simulation, VLN, and auxiliary vision-language data). Weights and training code promised open-source.

At release, GigaBrain-0.7 achieved substantial improvements over the preceding GigaBrain-0 series and prior SoTA (π₀.₅) on foundation zero-shot capability, language-conditioned instruction following, and post-training task success. In-house Maker H01 humanoid and mainstream robot embodiments both showed strong task adaptability across home and industrial scenarios.

---

## The three-system decomposition

```
      ┌──────────────────────────────────────────────────┐
      │   System 1: Understanding (VLM backbone)         │
      │   image + language → grounded scene tokens        │
      └──────────────────────────────────────────────────┘
                        │  (shared representation)
                        ▼
      ┌──────────────────┐        ┌──────────────────────┐
      │  System 2:       │        │  System 3:           │
      │  Prediction      │        │  Action              │
      │  (world model)   │        │  Generation          │
      │                  │        │                      │
      │  scene + action  │        │  scene + task →      │
      │  → future scene  │        │  action tokens       │
      └──────────────────┘        └──────────────────────┘
```

- **System 1** produces grounded scene tokens from image + language; this is the VLM stack, reused from prior GigaBrain-0 work.
- **System 2** conditions on scene tokens and a candidate action, predicts the future scene. Trained on paired (before, after) frames from the pretraining data.
- **System 3** consumes scene tokens plus a task specification, produces action tokens (unified across embodiments — see below).

The three systems share their input representation (System 1's grounded scene tokens) and gradients flow through all three during training. This is the "one-stage" property: no frozen-VLM-then-train-head cascade.

See [three-system-vla](../multimodal/three-system-vla.md) for the full mechanism.

---

## Data at a glance

```
Total pretraining:  37,000+ hours of heterogeneous embodied data
Composition (approximate, from paper description):
  ├─ Human egocentric video
  ├─ Robot trajectories across many embodiments (manipulation + navigation)
  ├─ Simulation
  ├─ Vision-language navigation (VLN)
  └─ Auxiliary vision-language corpora
```

Two design decisions matter here:

- **Heterogeneous by construction.** The prior GigaBrain-0 was already large; scaling further required leaving the trap of one-embodiment-per-model. Data from many embodiments (and human egocentric video, which has no explicit "action" annotation) is used jointly.
- **Multi-embodiment action generation.** Because System 3 covers many embodiments, action tokens must encode "which embodiment" as well as "what action". The paper's alignment training jointly optimizes for both understanding correctness (System 1) and action correctness across embodiments (System 3).

---

## Training recipe

### One-stage alignment

Instead of the standard cascade:

```
  freeze VLM → attach action head → train action head only
```

GigaBrain-0.7 does:

```
  train (VLM + prediction + action) jointly
    with a multi-objective loss covering all three systems
```

This is what "one-stage alignment" refers to. Two implications:

- **VLM comprehension co-adapts with action.** System 1's representations shift under gradients from System 3's action-correctness loss — the VLM learns to encode features action generation actually uses.
- **Prediction anchors representation.** System 2's world-model loss forces the shared scene representation to be predictive of future scenes, not just descriptive of current ones. Prior work on world-model auxiliary losses shows this improves policy quality; GigaBrain-0.7 folds it in from the start.

### Scale

37,000+ hours of pretraining data, model size not explicitly disclosed in the abstract but "foundation model" implied (multi-B scale). Weights promised open-source alongside training code.

---

## Post-training

The abstract emphasizes "post-training task success rates" separately from zero-shot capability, indicating that GigaBrain-0.7 goes through downstream task-specific post-training after the one-stage alignment. Details not given in the abstract; the release should clarify the exact recipe.

---

## Key results

- **Zero-shot capability**: substantial improvements over GigaBrain-0 and π₀.₅.
- **Language-conditioned instruction following**: substantial improvements.
- **Post-training task success**: substantial improvements.
- **Deployment**: strong task adaptability and completion on the Maker H01 humanoid across home + industrial scenarios; also mainstream robot embodiments.

Exact benchmark numbers await the full paper; the abstract states "substantial improvements" across all three axes.

---

## Why this matters

- **The scaling story for VLAs.** LLMs got emergent capability at pretraining scale; the open question for VLAs has been whether the same holds for embodied data. GigaBrain-0.7 is the largest single-system evidence point.
- **One-stage alignment as a design principle.** The dominant open recipe (freeze VLM, attach action head) trades a stable VLM for a rigid representation. One-stage alignment inverts that tradeoff — a real design principle that other VLA teams will need to evaluate.
- **Multi-embodiment as default, not an afterthought.** By training System 3 across embodiments from the start, GigaBrain-0.7 avoids the "one-embodiment-per-model" scaling ceiling that has held prior open VLAs back.

---

## Related concepts

- [three-system-vla](../multimodal/three-system-vla.md) — the specific understanding/prediction/action decomposition with one-stage joint training.
- [_vla](../multimodal/_vla.md) — taxonomy of vision-language-action approaches (RT-2, OpenVLA, π₀, GigaBrain).
- [qwen2-5 case study](qwen2-5.md) — the VLM-backbone stack GigaBrain-style systems build on.

## Sources

- Paper: *GigaBrain-0.7: Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture* — GigaBrain Team, 2026. [arXiv:2608.15875](https://arxiv.org/abs/2608.15875).
- Related: prior VLA work (π₀.₅, OpenVLA, RT-2) referenced as the baselines GigaBrain-0.7 surpasses.
