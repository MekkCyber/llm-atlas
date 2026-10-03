# pass@k
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An LLM evaluation metric that reports the probability that *at least one* of $k$ independent samples from the model is correct. pass@1 measures raw sampling quality; pass@K (K ≥ 2) measures the model's *solution coverage* — whether a correct answer exists in the sampled set at all. The gap between the two is a widely-used diagnostic for post-training sharpening and for test-time compute scaling.

**Prereqs:** [humaneval.md](humaneval.md) (origin benchmark)
**Related:** [../post-training/sharpening-tax.md](../post-training/sharpening-tax.md) · [../post-training/ptgs.md](../post-training/ptgs.md) · [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

For a problem with a known correctness check (unit test, math-answer match, verifier), sample $n$ independent completions from the model, count $c$ that pass. The unbiased estimator for pass@$k$ (with $k \le n$) is:

$$
\mathrm{pass@}k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}
$$

Introduced in the HumanEval paper (Chen et al., 2021) to measure the probability that one of $k$ sampled solutions passes the tests. Now used across code, math, agentic, and tool-use benchmarks whenever there is a reliable per-sample verifier.

## How it works

- Choose a target $k$: 1, 5, 10, K-of-interest.
- Sample $n \ge k$ completions per prompt at a fixed temperature (often $T=0.8$ or $T=1.0$; pass@1 is sometimes reported at $T=0$ as "greedy pass@1" — a different quantity).
- Compute the estimator per prompt and average over prompts.

Three regimes:

- **pass@1** approximates expected per-sample quality. The quantity leaderboards usually report.
- **pass@K** (K large) approximates *solution coverage* — the probability that the model's output distribution contains a solution anywhere.
- **(pass@K − pass@1)** is a diagnostic of the model's exploration vs exploitation balance. A sharp policy has low coverage overhead; a diverse base model has large.

## Why it matters

- **Separates "the model can" from "the model does."** Many RL-post-trained models have better pass@1 than their base but *worse* pass@K: they concentrate probability on a subset of solutions and lose others. The *Sharpening Tax* metric formalizes this gap (2026).
- **Right metric for test-time-compute recipes.** Majority-voting, best-of-N, verifier-reranked best-of-N, agentic retry — they all consume samples. pass@K sets the ceiling for what any such recipe can reach.
- **Compatible with any rule-verifier benchmark.** HumanEval, MBPP+, LiveCodeBench, MATH-500, AIME, SWE-Bench Verified, Terminal-Bench, GAIA.

## Gotchas & tricks

- **Temperature choice matters.** pass@1 at $T=0$ (greedy) and pass@1 estimated from samples at $T=0.8$ are different numbers; many papers quietly report one and label the other.
- **$n$ should be ≫ $k$** for the estimator's variance to be small; typical setups are $n=20$–$200$ for $k \in \{1, 5, 10\}$.
- **Correlated failures bias pass@K upward.** If samples share failure modes (same wrong-but-plausible lemma), coverage is overestimated vs truly independent sampling.
- **Agentic pass@K is per-trajectory, not per-token.** Each independent *run* of the agent (with its own tool calls) counts as one sample. Expensive to estimate at large $k$ for long-horizon tasks.
- **pass@K is not an upper bound on real-user utility** — a user can't afford $K$ attempts; pair pass@K with cost-adjusted metrics (pass@1-of-N-at-fixed-token-budget).

## Sources

- Paper: *Evaluating Large Language Models Trained on Code* (HumanEval) — Chen et al., OpenAI, 2021 — defines pass@k with the unbiased estimator.
- Paper: *Sharpening Tax in Post-Training* — Oh et al., 2026 — [arXiv:2610.01509](https://arxiv.org/abs/2610.01509). Operationalizes (pass@K − pass@1) as a cost metric across post-training.
- Paper: *Large Language Monkeys* — Brown et al., 2024 — large-$K$ scaling study on code and reasoning benchmarks.
