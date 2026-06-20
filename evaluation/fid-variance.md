# FID Variance
*Depth — treating FID as a random variable over training and sampling seeds, and quantifying how much of any reported FID delta is signal vs. seed lottery.*

**TL;DR:** "The FID Lottery" (Dufour, Efros, Pérez 2026) trains *several hundred* SiT networks on class-conditional ImageNet 256² and measures FID on a two-axis panel of training seeds × generation seeds. Findings: **retraining moves FID 3.2× more than resampling** (in Inception feature space); the spread does not shrink with compute or model size; per-cell CFG tuning halves variance but reshuffles which seeds win; a lucky training seed reaches the same FID with up to 2× less compute than an unlucky one.

**Prereqs:** *(none — basic familiarity with FID helps)*
**Related:** *(none — diffusion / generative-model depth files not yet in the graph)*

---

## What it is

The FID (Fréchet Inception Distance) is reported as a single scalar in nearly every diffusion / flow paper. It depends on at least two random choices — the **training seed** (controls init, data ordering, per-step flow-matching noise) and the **generation seed** (controls the sampling SDE). Single-seed FID conflates these and reports a point estimate whose noise floor is usually not estimated. This paper directly measures the noise floor and breaks it down.

## How it works

Two-axis seed panel:

```
                 +------------- training seeds (s_train) -------------+
                 |                                                    |
generation       |    FID(s_train, s_gen)                             |
 seeds (s_gen)   |    measured for ~hundreds of (s_train, s_gen) cells |
                 |                                                    |
                 +----------------------------------------------------+
```

Each cell is a fully trained SiT model (class-conditional, ImageNet 256²) evaluated under a specific generation seed. The paper then decomposes total variance into:

- **Initialization** (different random init).
- **Data ordering** (different shuffle of the training corpus).
- **Per-step Gaussian noise** of the flow-matching loss.
- **Generation noise** (the sampling SDE seed).

Per-cell CFG tuning is treated as an additional axis — the optimal CFG scale itself varies across training seeds.

## Why it matters

- Most reported FID deltas in diffusion / flow papers are **below the seed-noise floor** the paper measures. Many "improvements" are seed luck.
- Gives the community a concrete CoV band (~1–2%) to compare against — improvements smaller than this band require multiple-seed reporting to be credible.
- The compute-equivalence finding (lucky seed = 2× cheaper) implies that **single-seed FID papers systematically over-report compute efficiency** — the lucky-seed paper looks more sample-efficient than its method actually is.
- Template for similar variance audits of LLM benchmarks: how much of a 1-point MMLU gap is real?

## Gotchas & tricks

- Per-cell CFG tuning halves spread *and reshuffles winners*. Reporting "best seed with best CFG" is the worst combination — you've selected on the noise.
- Inception-feature-space distance is the right unit for comparing variance magnitudes; pixel-space comparisons mislead.
- The 1–2% CoV is specific to SiT + ImageNet 256² + flow-matching loss. Architecture / loss / scale changes could shift it; the methodology generalizes, the constants don't.
- Doesn't argue FID is *useless* — argues single-seed FID is. Mean ± std over a small seed panel is the cheap fix; the paper provides protocol guidance.
- Distinct from *reproducibility* (can someone else's run match yours): variance studies acknowledge irreducible noise inside one team's own runs.

## Sources

- Paper: *The FID Lottery: Quantifying Hidden Randomness in Generative Model Evaluation* — Dufour, Efros, Pérez, Kyutai / UC Berkeley, 2026 — arXiv 2606.20536.
- Predecessor framing: *The Lottery Ticket Hypothesis* — Frankel & Carbin, 2019 — uses the same "lottery" metaphor for init sensitivity.
- Reference for FID: Heusel et al. 2017.
