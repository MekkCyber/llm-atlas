# Few-step diffusion distillation (recipe-centric)
*Depth — one specific technique area, grounded in its source paper(s).*

**TL;DR:** Few-step distillation accelerates large image diffusion models from dozens of denoising steps down to 1–8. Prior work focused on the *objective* (consistency loss, progressive distillation, score distillation). Qwen-Image-Flash (2026) argues the *training recipe* — data composition, teacher guidance schedule, and task mixture — matters as much as the objective, and demonstrates that a strong objective with a poor recipe under-performs a moderate objective with a good recipe.

**Prereqs:** [README.md](README.md)
**Related:** (no existing diffusion pages in the graph — this is a new branch)

---

## What it is

A modern image generator is typically a 50- to 100-step iterative denoiser (DiT/UNet). Few-step distillation trains a *student* model that approximates the teacher's denoising trajectory in a small number of steps. The student is initialized from the teacher and optimized to match the teacher's outputs (per-step or via the final image) with various consistency / score / adversarial loss formulations.

Qwen-Image-Flash zooms out from the loss formulation to ask: *given a fixed loss, what training recipe makes the student best?*

## How it works

The recipe-centric framing identifies three knobs:

1. **Data composition.** The teacher's training data may not match the desired student distribution. Mixing extra domains during distillation (text-rich images, edits, photo vs illustration) systematically changes the student's behavior in domains *not* directly distilled — including degrading them.
2. **Teacher guidance.** Classifier-free guidance scale at teacher-sampling time matters more than the literature suggests. The optimal CFG schedule for distilling T2I differs from instruction-guided editing; a fixed CFG hurts the under-served task.
3. **Task mixture.** When distilling a multi-task teacher (T2I + image editing), the per-task mixing ratio is a first-order choice. Under-mixing one task can degrade *both* tasks, not just the under-mixed one — apparently because shared layers benefit from the diversity of supervision.

The paper holds the consistency-style distillation objective fixed and sweeps these three axes on Qwen-Image-2.0 as the teacher, producing the Qwen-Image-Flash student.

## Why it matters

- **Production-relevant.** Few-step diffusion is the dominant pattern in shipped image-gen products (Imagen Lightning, FLUX-Schnell-class, SDXL-Turbo). Recipe-level lessons transfer directly to anyone training derivatives.
- **Corrects a methodology gap.** Distillation papers commonly report objective-only ablations, which misleads readers who try to replicate without the recipe. This work explicitly separates loss from recipe.
- **Non-obvious findings.** Task mixtures that look balanced (50/50) often aren't optimal; CFG schedules that match the teacher's training regime aren't best for the student.

## Gotchas & tricks

- **Recipe is teacher-specific.** Optimal data composition and task mix depend on the teacher's training distribution. Transferring a recipe verbatim across teacher families is unsafe.
- **Watch for cross-task interference.** Distilling editing alongside T2I can improve both, but only at the right mix; off-mix hurts the harder task disproportionately.
- **Eval downstream tasks.** Few-step distillation that scores well on FID may degrade specific applications (text rendering, fine detail). Evaluate on the actual workload.
- **Step count and recipe interact.** A recipe tuned for 4-step distillation may not be optimal for 1-step. Re-tune.

## Sources

- Paper: *Qwen-Image-Flash: Beyond Objective Design* — Wu et al., 2026 — [arXiv:2606.03746](https://arxiv.org/abs/2606.03746).
- Related: consistency models (Song et al., 2023); progressive distillation (Salimans & Ho, 2022).
