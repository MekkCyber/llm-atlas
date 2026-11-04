# History Forcing
*Depth — long-horizon video-diffusion distillation that amplifies the history-dependent component of the teacher's prediction.*

**TL;DR:** Autoregressive video generators drift over long horizons — details change on revisit, motion becomes incoherent. **History Forcing** (ZJU / MSR, 2026) is a distillation objective for few-step video students that compares teacher predictions *with* and *without* historical context, weights the difference by a scalar `λ`, and adds it to the student's target. This separates "what the frame looks like" from "what the frame looks like given history," so the student can inherit long-range consistency without the teacher's per-frame visual artifacts. Reaches VBench 85.07 at 15 s and VBench-Long 79.82 at 60 s with 1.3B models.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md) · [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [wave-forcing.md](./wave-forcing.md)

---

## What it is

A distillation objective for **few-step** (≤ 8-step) autoregressive video diffusion students, aimed at long-horizon consistency rather than per-frame quality. Standard distillation uses the teacher's full prediction as a soft target; History Forcing decomposes that target into a history-free part and a history-delta part and amplifies the latter.

Two application regimes:
- **Long-video generation.** History is previously-generated chunks from the student's own rollout. Teaches the student to attend to its own earlier output coherently.
- **World-model consistency / out-of-sight.** History is a fixed-view teacher rollout (what the world looked like before the camera moved away). Teaches the student to produce consistent revisits.

## How it works

For each frame-chunk `x_i` the student produces with history `h_i`:

1. Run the teacher twice:
   - `ŷ_full = teacher(x_i, h_i)` — prediction with history.
   - `ŷ_nohist = teacher(x_i, ∅)` — prediction with history masked out.
2. Form the distillation target:

   ```
   target = ŷ_nohist + λ · (ŷ_full − ŷ_nohist)
   ```

   `λ > 1` amplifies the history-dependent component; `λ = 1` recovers plain distillation; `λ < 1` dampens it.

3. Standard distillation loss between the student's prediction and `target`.

Because the delta `ŷ_full − ŷ_nohist` is defined entirely by what history contributes, the student learns history-dependent behavior independently of the teacher's visual quality — the no-history part cancels. With `λ > 1`, the student is explicitly taught that "revisiting an object should look like what you saw before" more sharply than the teacher demonstrates on its own.

## Why it matters

- **Attacks the main video-WM failure mode.** Long-horizon drift and inconsistency on revisit are the dominant failure modes for current video world models used by agents. History Forcing is a distillation-stage fix that doesn't require architecture or data changes.
- **Decouples quality axes.** Separates "frame looks good" from "frames remain coherent" — typically tangled in standard distillation.
- **Cheap.** Two teacher forward passes per chunk vs one; quadratic blow-up in context length can be managed by caching.

## Gotchas & tricks

- **`λ` tuning.** Too large and the student overweights history and produces static scenes; too small and drift persists. The paper's `λ` sweep lands between 1.5 and 2.5 for most settings.
- **Not a quality upgrade.** If the teacher is already weak on per-frame quality, History Forcing doesn't fix it — the no-history part still looks bad; only the delta is reshaped.
- **Chunk boundary matters.** The method operates on chunks; attention across chunk boundaries needs to see history coherently or the delta is noise.
- **Feedback-loop risk during self-generation long-video.** If the student's own history is low-quality, training on amplified-delta over bad history can amplify bad patterns. Use teacher-forced history during early training.

## Sources

- Paper: *History Forcing: Distilling History-Dependent Continuations for Long-Horizon Video World Models* — Kuang, Wang et al., ZJU / Microsoft Research / Monash / Adelaide, 2026 — [arXiv:2611.02782](https://arxiv.org/abs/2611.02782).
- Benchmark: VBench, VBench-Long — long-video evaluation suites referenced in the results.
