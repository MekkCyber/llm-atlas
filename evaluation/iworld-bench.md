# iWorld-Bench
*Depth — a benchmark for long-horizon interactive video-world generation, released with AlayaWorld.*

**TL;DR:** A benchmark introduced alongside AlayaWorld for evaluating **long-horizon** performance of interactive video world models — the regime where drift, forgetting, and camera-return-failure dominate quality. AlayaWorld reports the best performance on iWorld-Bench; the interesting piece is the benchmark itself as the first shared testbed for the video-world-model family.

**Prereqs:** *(none)*
**Related:** [../case-studies/alayaworld.md](../case-studies/alayaworld.md), [../multimodal/bounded-visual-context.md](../multimodal/bounded-visual-context.md)

---

## What it is

Video-world-model quality has mostly been measured on short-clip metrics (FVD-style perceptual scores, standard video benchmarks) — but the point of a *world* model is long-horizon consistency, camera-loop closure, prompt-switching stability, and interactive controllability. Short-clip benchmarks don't measure any of those.

iWorld-Bench targets the long-horizon regime specifically — the regime where all the interesting world-model failure modes live.

## How it works

Reported details from the release paper are limited to the framing (long-horizon interactive video-world generation) and to AlayaWorld's own headline result (best on the benchmark). Concretely, expect the benchmark to include:

- **Long-horizon rollouts** — multi-minute video sequences beyond typical short-clip video benchmarks.
- **Interactive controllability probes** — prompt-switching, camera-trajectory-following, seed-conditioning.
- **Consistency metrics** — scene persistence, camera-return closure, identity preservation.

The paper positions iWorld-Bench as an open, extensible testbed for the video-world-model family, matching AlayaWorld's "full-stack, open-source, long-term project" framing.

## Why it matters

- **Fills a real evaluation gap.** Every video-world-model paper on HF/arXiv this quarter reported per-paper metrics; iWorld-Bench is a shared testbed to compare across them.
- **Long-horizon focus.** Redirects benchmark pressure to the actually-hard regime rather than the easy short-clip one.
- **Concurrent with a wave.** Released the same day as ABot-World-0 (2607.19191), AlayaRenderer-Flash (2607.18703), and Masked Visual Actions (2607.19343) — expect subsequent releases to report iWorld-Bench scores to be comparable.

## Gotchas & tricks

- Details of the exact metric definitions are not in the abstract-only description; check the paper before reproducing.
- "Best on benchmark" from the releasing team is always the weakest comparison point; independent evaluations are what will make iWorld-Bench a real standard or not.

## Sources

- Paper: *AlayaWorld: Interactive Long-Horizon World Modeling* — Zhang, Li, Zhan, Ge, Yin et al. (Alaya Lab), 2026 — [arXiv:2607.18367](https://arxiv.org/abs/2607.18367)
