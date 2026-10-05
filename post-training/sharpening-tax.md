# Sharpening Tax
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A diagnostic metric for post-training that quantifies how much *solution coverage* (pass@K) a model loses while gaining *single-shot accuracy* (pass@1). RL post-training pushes tasks toward "always solved" or "never solved" and improves pass@1 at the cost of pass@K. Companion method **PTGS** adapts sampling temperature per prompt to its estimated difficulty to pay a smaller tax.

**Prereqs:** [_rl.md](./_rl.md), [rlvr.md](./rlvr.md)
**Related:** [grpo.md](./grpo.md), [rejection-sampling.md](./rejection-sampling.md), [../evaluation/aime.md](../evaluation/aime.md)

---

## What it is

Oh et al. (Wisconsin / Stanford / Google, 2026) formalise the "RL sharpens, it doesn't teach new skills" hypothesis as a measurable quantity. Across 14 base/post-trained model pairs from four families on three agentic benchmarks (42 cases total), they find that pre-trained bases paired with a light inference harness often **beat** their post-trained counterparts in pass@K given enough test-time budget — despite far lower pass@1.

## How it works

**The metric.** For each (base, post-trained) model pair and each benchmark, measure pass@K for a range of $K$. The **Sharpening Tax** is the gap between the base model's pass@K curve and the post-trained model's pass@K curve — specifically, how much coverage was lost in exchange for the single-shot improvement at $K=1$. Estimable from a few rollouts; correlates well with other coverage/entropy metrics.

**Mechanism.** Post-training's effect on the per-prompt success-rate distribution is bimodal: prompts drift toward either $\approx 0$ or $\approx 1$ success rate, cutting the long middle tail that pass@K sampling depends on.

**PTGS (Posterior-Tempered Group Sampling).** A plug-and-play Bayesian sampler: for each prompt, estimate its difficulty from early rollouts and set the sampling temperature from the posterior. Easy prompts get low temperature (exploit); hard prompts get high temperature (explore). Applied during RL training — not at inference — so the RL update sees rollouts spanning both regimes.

## Why it matters

- Rehabilitates "base model + inference harness" as a serious baseline for agentic eval. Many published RL gains disappear under pass@K.
- Gives a single number to flag "this RL run is actively narrowing the capability frontier" — useful during RL recipe development.
- PTGS integrates with existing GRPO/RLVR trainers without changing the policy objective: it is a sampler, not a new loss.
- In the paper's two agentic environments, PTGS solved more tasks under repeated sampling **and** improved pass@1 vs the fixed-temperature baseline — the usual pass@1 ↔ pass@K tradeoff is a sampling-regime artifact.

## Gotchas & tricks

- Measuring the tax requires enough rollouts per prompt to estimate pass@K reliably at the $K$ values of interest; typical ranges are $K \in \{1, 2, 5, 10, 50\}$.
- The tax is not uniformly bad — on tasks where a wrong answer is costly (safety, consequential actions), a model that converges to one answer is actually what you want.
- PTGS's difficulty estimate uses early rollouts, so the per-prompt budget must be at least a few samples before the posterior is informative.
- The paper's benchmarks are agentic; mapping to pure math-reasoning (AIME-style, pass@1 is the norm) is less clear.

## Sources

- Paper: *Sharpening Tax in Post-Training* — Oh, Zeng, Qi, Zhmoginov, Lei, He, Phan, Kang, Mirhoseini, Li; Wisconsin / Stanford / Google, 2026 — [arXiv:2610.01509](https://arxiv.org/abs/2610.01509).
- Context: *pass@K* — Chen et al., 2021 (Codex) — origin of the sampling-based coverage metric this paper repurposes as a post-training diagnostic.
