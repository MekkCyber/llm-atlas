# Three-System VLA Architecture
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Prior vision-language-action models bolt a small action head onto a frozen VLM. **GigaBrain-0.7** proposes decomposing the VLA into **three coupled subsystems** — understanding (VLM), prediction (world model), action (multi-embodiment controller) — sharing a scene representation and trained jointly via **one-stage alignment**, so understanding, prediction, and control co-adapt rather than compose. Introduced by the GigaBrain Team, 2026.

**Prereqs:** [README.md](README.md), [_vla.md](_vla.md)
**Related:** [../case-studies/gigabrain-07.md](../case-studies/gigabrain-07.md)

---

## What it is

Two things are wrong with the dominant open-VLA recipe:

1. **Frozen VLM constrains action.** With the VLM's representation fixed, the action head must fit within whatever features the VLM already encodes. If those features don't include what action decoding needs (fine-grained affordances, precise 3D geometry, temporal consistency), the action head can't compensate.
2. **Two-stage cascades leave capabilities unshared.** After the action head is trained, the VLM's original tasks (VQA, captioning) still work, but action-relevant improvements the VLM *could* make don't transfer back.

Three-system VLA fixes both by treating the system as three subsystems from the start.

## How it works

### The three subsystems

```
                 Vision-Language Input
                          │
                          ▼
              ┌─────────────────────────┐
              │   System 1: VLM backbone │
              │   → grounded scene       │
              │     representation       │
              └─────────────────────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
    ┌────────────────┐        ┌──────────────────┐
    │ System 2:      │        │ System 3:        │
    │ World          │        │ Action           │
    │ Prediction     │        │ Generation       │
    │                │        │ (multi-embodi-   │
    │ scene + action │        │  ment)           │
    │ → future scene │        │ scene + task →   │
    │                │        │ action tokens    │
    └────────────────┘        └──────────────────┘
```

- **System 1** is the shared VLM backbone. Produces grounded scene tokens.
- **System 2** takes scene tokens plus a candidate action; predicts the *future* scene tokens. World-model auxiliary loss.
- **System 3** takes scene tokens plus a task specification; produces action tokens for the target embodiment.

The three systems share System 1's output representation. During training, gradients from Systems 2 and 3 flow into System 1 — so the shared representation adapts to what prediction and action need.

### One-stage alignment

Rather than the cascade "pretrain VLM → freeze → train action head", one-stage alignment trains all three subsystems jointly on the pretraining data:

$$
L_{\text{total}} = \lambda_1 \cdot L_{\text{understand}} + \lambda_2 \cdot L_{\text{predict}} + \lambda_3 \cdot L_{\text{act}}
$$

with multi-embodiment action loss that jointly covers all embodiments in the training data.

### Multi-embodiment action

System 3 takes both a task specification *and* an embodiment identifier; produces action tokens in that embodiment's action space. Because training data spans multiple embodiments, System 3 learns the shared action-space structure (temporal smoothness, task-conditioned reachability) plus embodiment-specific decoding.

## Why it matters

- **Substantial gains over frozen-VLM baselines.** Compared to prior GigaBrain-0 series and π₀.₅, GigaBrain-0.7 shows substantial improvements in zero-shot capability, language-conditioned instruction following, and post-training task success.
- **One design principle, three payoffs.** Coupling the three systems gives: (1) an action-informed VLM representation, (2) a world-model auxiliary loss that anchors representation quality, (3) multi-embodiment coverage without per-embodiment fine-tuning.
- **A repeatable architectural pattern.** Any VLA follow-up can adopt three-system decomposition and one-stage alignment without reinventing them. The design principle generalizes to VLA sizes and embodiments outside GigaBrain-0.7's specific setup.

## Gotchas & tricks

- **Loss balancing is a real hyperparameter.** With three losses, the mixing weights $\lambda_i$ trade off understanding quality, prediction sharpness, and action correctness. Early training will need loss-scale monitoring per subsystem.
- **World-model loss can dominate.** Prediction losses over pixel/token sequences are large in magnitude; if not down-weighted, System 2 can overwhelm System 3's gradient signal. Normalize per-loss or use gradient-scale balancing.
- **Multi-embodiment action space needs a shared representation the model can learn.** Different embodiments have wildly different joint spaces; naïve concatenation of embodiment-specific channels doesn't generalize. Consider unified 6D-pose parameterization or embodiment-conditioned action tokenizers.
- **World-model quality is asymmetric.** System 2 is easier to train well on data where actions have visible consequences (manipulation) and harder on human egocentric video (no controllable action). Reweight per data source.
- **Frozen-VLM baselines are still cheaper.** If your embodiment coverage is narrow and your VLM already handles the visual tasks well, the cascade is faster to train. Three-system pays off at scale and across embodiments.

## Sources

- Paper: *GigaBrain-0.7: Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture* — GigaBrain Team, 2026. [arXiv:2608.15875](https://arxiv.org/abs/2608.15875).
- Related: prior GigaBrain-0 series — earlier iteration without three-system decomposition.
- Related: *π₀.₅* — the frozen-VLM baseline GigaBrain-0.7 surpasses.
