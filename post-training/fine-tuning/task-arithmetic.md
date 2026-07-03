# Task Arithmetic
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Fine-tuning a model on a task produces a **task vector** $\tau = \theta_{\text{FT}} - \theta_0$ in weight space. These vectors compose surprisingly linearly: **adding** a task vector installs the behaviour, **negating** it removes the behaviour, and *analogies* work — $\tau_{\text{A}\to\text{B}} + \theta_{\text{C}}$ transfers B onto C. First shown for CLIP / GPT-2 (Ilharco 2022); recently extended to VLAs via **DART** (2026), which uses one demo to compute a shift vector and adapts a VLA to a new camera pose / robot in one shot.

**Prereqs:** [README](README.md), [../../pre-training/model-souping.md](../../pre-training/model-souping.md)
**Related:** [../../architectures/transformer-block.md](../../architectures/transformer-block.md)

---

## What it is

Task arithmetic treats a fine-tuned model as *the base model plus a direction in parameter space*:

$$
\tau_{\text{T}} \;=\; \theta_{\text{FT-on-T}} - \theta_0
$$

The vector $\tau_T$ is a concrete artifact — same shape as the model weights. Empirically, three operations on task vectors preserve semantics:

- **Addition:** $\theta_0 + \tau_T$ ≈ $\theta_{\text{FT-on-T}}$ — obvious, but also $\theta_0 + \sum_i \tau_{T_i}$ approximately learns all tasks jointly (multi-task without joint training).
- **Negation:** $\theta_0 - \tau_T$ *removes* T-capability from the base — used for unlearning, safety-behaviour subtraction.
- **Analogy:** $\tau_{A\to B}$ + $\theta_C$ transfers the A→B transformation onto a different base — the DART use case.

## How it works

The operation is trivial (element-wise vector arithmetic on weight tensors). The interesting content is *when* it works.

### Base recipe

1. Fine-tune $\theta_0 \to \theta_T$ on task $T$ (SFT, LoRA-merged, or any full-parameter method).
2. Store $\tau_T = \theta_T - \theta_0$.
3. To apply to a new base $\theta_0'$: $\theta_0' + \alpha \tau_T$ with $\alpha \in [0.5, 1.0]$ typical.

Linearity is imperfect: interference between task vectors grows with parameter overlap. Common mitigations: **TIES-merging** (trim small-magnitude entries, elect signs), **DARE** (drop-and-rescale before merging), **task-vector orthogonalisation** during fine-tuning.

### DART (2026): weight arithmetic for VLAs

VLA adaptation under **environmental shifts** (camera pose, related robot) is normally per-task demonstration-heavy. DART treats the environmental shift as a domain whose signal factors into a weight-space direction:

1. Fine-tune on **one demonstration** in the shifted environment → $\theta_{\text{shift}}$.
2. Compute $\tau_{\text{shift}} = \theta_{\text{shift}} - \theta_0$.
3. Add $\tau_{\text{shift}}$ to the base VLA and evaluate on many tasks in the shifted env.

The evidence: one-shot success rates match or approach multi-demonstration per-task fine-tuning.

## Why it matters

- **Composition without retraining.** Multi-task deployment reduces to storing $\tau$s per task, not per-combination fine-tuned checkpoints.
- **Unlearning / safety subtraction.** Task-vector negation is one of the cleaner unlearning tools — remove capability by subtracting its direction.
- **Cheap domain shift for large models.** DART shows it transfers to VLAs — a class of models where full retraining costs are prohibitive. If the pattern generalises across foundation-model families, per-environment adaptation stops being a training problem.

## Gotchas & tricks

- Linearity is *approximate*. Adding many task vectors accumulates interference; TIES / DARE / RegMean-style tricks recover most of it up to ~10 tasks.
- Coefficient $\alpha$ matters. $\alpha = 1.0$ can overshoot; grid-search on a small held-out is standard.
- The base model must be shared. Task vectors do not transfer between different pretrained bases — mismatched initialisations kill the semantics.
- LoRA task vectors compose better than full-parameter task vectors: the low-rank structure limits interference.
- For VLAs (DART): the technique works for *environmental* shifts (camera, similar robot). It does not compose across **different embodiments** — the shift direction becomes semantically meaningless there.
- Task-vector negation for unlearning is measurable but not adversarially safe — a capable adversary can undo the subtraction with more fine-tuning.

## Sources

- Paper: *Editing Models with Task Arithmetic* — Ilharco et al., 2022 — the original task-vector formalism (CLIP, GPT-2).
- Paper: *TIES-Merging: Resolving Interference When Merging Models* — Yadav et al., 2023 — multi-task task-vector interference mitigation.
- Paper: *Domain Arithmetic: One-Shot VLA Adaptation under Environmental Shifts (DART)* — Kim, Shin, Choi, 2026 — [arXiv:2607.00666](https://arxiv.org/abs/2607.00666).
- Paper: *Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time* — Wortsman et al., 2022 — sibling weight-averaging line.
