# MT Forcing (Multi-Turn Image Editing)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training recipe for **autoregressive diffusion** image editors that attacks the drift and quality degradation that compound across multiple edit turns. Two components: **Single-Turn DMD** — a teacher that applies the whole cumulative edit sequence in one step guides the student's self-generated rollouts — and **Teacher-Forcing Guidance** — train on consistent reference trajectories first as warmup, then as a regulariser during distillation. On SenseNova-U1 / U1.5 the recipe improves instruction following, cross-turn content consistency, and visual stability, with the largest wins in later turns of the edit chain.

**Prereqs:** [attention](../fundamentals/attention.md), [_rl](../post-training/_rl.md)
**Related:** [grpo](../post-training/grpo.md)

---

## What it is

A unified multimodal model that does image editing autoregressively — one edit per turn, each conditioned on the running image — accumulates two kinds of drift: *content drift* (unintended changes to parts the instruction didn't touch) and *quality drift* (artifacts that compound). The root cause is **exposure bias**: at training time the model sees reference histories, but at inference time it must condition on its own (imperfect) prior outputs. By the fourth turn, the input distribution has drifted far from training.

MT Forcing attacks this by shaping the training distribution to look like the inference distribution — the standard exposure-bias playbook — but specialised for AR-diffusion image editing.

## How it works

**Component 1: Single-Turn DMD.** DMD (Distribution Matching Distillation) is a one-step distillation technique for diffusion models — the student learns to match the teacher's distribution in one step rather than many. MT Forcing's twist: the *teacher* applies the full cumulative edit sequence (turn 1 → turn N) to the source image in a *single* step; the student rolls out its own multi-turn trajectory; the student is distilled toward the teacher's final image. This gives a trajectory-independent target: no matter which path the student took, the ground-truth final state is the same.

**Component 2: Teacher-Forcing Guidance (TFG).** During distillation, the student is additionally regularised to stay close to a *consistent reference trajectory* — one in which every intermediate image is the one the teacher would have produced after that prefix of edits. TFG is applied in two phases:
- **Warmup.** Pure teacher-forcing training on reference trajectories, so the student has a stable starting point.
- **Distillation phase.** TFG becomes a soft regulariser during the Single-Turn DMD loop, pulling the student back when its rollouts diverge from the reference.

The combination means the student sees training signal that spans both *trajectory-independent* final-image targets (DMD) and *consistency* targets (TFG), covering the two failure modes empirically.

## Why it matters

- **Interactive editing is the production use case.** Single-shot generation is already strong; the actual bottleneck for creative tools is multi-turn stability.
- **Drop-in training recipe.** Both components attach to existing AR-diffusion image-editor stacks without architectural changes. The authors demonstrate on SenseNova-U1 / U1.5.
- **Exposure bias framed as a distillation problem.** Reframing drift as a train/test distribution mismatch for distillation makes the two-component fix clean, rather than treating each drift mode as a separate hack.
- **The gain concentrates where you need it.** Later turns (3+) are where previous recipes fell apart; MT Forcing's largest wins are there.

## Gotchas & tricks

- **Teacher must be trusted for cumulative edits.** The Single-Turn DMD teacher *must* produce a believable "all edits applied at once" image. For edits with path-dependent semantics (an edit chain where turn 3 uses turn 2's output as source material), the single-shot teacher is a crude approximation.
- **Warmup length matters.** Insufficient teacher-forcing warmup leaves the student too far from the reference trajectory for TFG to regularise effectively. The paper tunes this per model.
- **DMD step count.** Single-step DMD is the paper's choice; multi-step DMD can improve absolute quality at the cost of the trajectory-independence property. Pick based on target latency.
- **Not for text generation.** The recipe is specific to image-editing AR-diffusion. The exposure-bias framing generalises, but the specific DMD + TFG combination is tuned for the pixel-level cumulative-edit setting.
- **Instruction-following vs. consistency can trade off.** Aggressive TFG regularisation suppresses large-edit instructions. The hyperparameter balance is model-specific.

## Sources

- Paper: *MT Forcing: Towards Stable and Consistent Multi-Turn Image Editing with Autoregressive Diffusion* — Chen, Wang, Fan, Qiu, Wei, Yang, Yang, Zhang, Yang, Zhao, Yang, Rao, 2026 — [arXiv:2610.25404](https://arxiv.org/abs/2610.25404).
- Background: *One-step Diffusion with Distribution Matching Distillation* — Yin et al., 2024 — introduces the DMD framework MT Forcing adapts.
