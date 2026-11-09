# Benchmark compression
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Running the full version of a benchmark on every candidate model is cost-prohibitive, especially for multimodal and agent suites. Benchmark compression evaluates on a carefully chosen *subset* whose score closely estimates the full-benchmark score. The surprising empirical finding (BestRand, 2611.00883): under modern foundation models, estimation error from a random subset is long-tailed — most samples are near-exact, but a few are badly off — and simply filtering the tail outperforms more elaborate adaptive methods.

**Prereqs:** [mmlu](mmlu.md)
**Related:** [livecodebench](livecodebench.md), [math500](math500.md), [decontamination](../data/decontamination.md)

---

## What it is

Given a benchmark $B = \{x_i, y_i\}_{i=1}^N$ with full-set score $S(M) = \frac{1}{N}\sum \mathbb{1}[M(x_i) = y_i]$, a benchmark-compression method selects a subset $B' \subset B$ with $|B'| = k \ll N$ and reports $\hat S(M) \approx S(M)$. The two quality criteria are (i) unbiasedness ($\mathbb{E}[\hat S] = S$) and (ii) low variance (small $|\hat S - S|$ with high probability).

Competitive methods include **Anchor Points**, **IRT selection**, and **stratified sampling**. BestRand is the modern random-sampling baseline with post-hoc tail control.

## How it works

The BestRand recipe:
1. Derive an analytic approximation to the estimation-error distribution for random $k$-subsets under a cohort of foundation models. The derivation shows the distribution is long-tailed.
2. Draw many candidate random subsets.
3. Score each candidate by a cheap filter (the paper's exact filter uses the analytic error bound; see the source).
4. Return the subset with the smallest predicted tail risk.

Why not just use random sampling? Random is unbiased *in expectation* but has non-trivial probability of a bad draw. Why not IRT-style adaptive selection? It costs a calibration model evaluation on the full set first, which defeats the compression purpose.

## Why it matters

- **70+ suites validated** — text, multimodal, agent, and VLA benchmarks.
- **50.2% reduction in high-error risk, 16.4% lower mean estimation error, 21.8% lower OOD error** (shifts across score ranges, model families, time) versus competitive methods — with only 51.3% of their construction cost.
- Makes it cheap for a lab to produce defensible leaderboard numbers without running the full suite on every candidate checkpoint.

## Gotchas & tricks

- The long-tail result depends on the model cohort being "modern foundation models" — the shape of the error distribution shifts for small/specialized models and the filter may need recalibrating.
- Compressed subsets should be **re-sampled** when the model cohort changes significantly; a 2023-calibrated subset may not predict 2026 model scores.
- Never use benchmark compression for *ranking* at the top of the leaderboard — the residual variance can flip adjacent models. Use for coarse screening.
- Report the full-set score alongside the compressed score whenever possible, with a confidence interval.

## Sources

- Paper: Random Is Almost Enough? Stable and Low-Cost Benchmark Compression via BestRand — Ling et al., 2026 — [arXiv:2611.00883](https://arxiv.org/abs/2611.00883)
- Code: https://github.com/George-Ling3/BestRand
