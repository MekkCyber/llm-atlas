# CausalDS
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for LLMs acting as **data-science agents** across all three rungs of **Pearl's causal hierarchy** — association, intervention, counterfactual. Each instance bundles a synthetic **structural causal model** (SCM), observational data drawn from it, and a domain-grounded narrative. Scoring is **deterministic**, treats **non-identifiable cases as first-class outcomes** (agents must decline, not fabricate), and evaluates five interlocking axes — **symbolic causal reasoning, data science, uncertainty quantification, abstention, and tool use** — jointly. Finding: frontier and open-weight models dissociate primarily on **epistemic axes** (abstention, uncertainty) rather than on symbolic reasoning. Introduced by Leban & Sun (U. Michigan Statistics), 2026 (arXiv 2607.08093).

**Prereqs:** *(none)*
**Related:** [uniclawbench.md](./uniclawbench.md) · [mmlu.md](./mmlu.md) · [ifeval.md](./ifeval.md)

---

## What it is

An agent benchmark for the intersection of *causal reasoning* and *data science*. Prior work split the two: symbolic causal-reasoning benchmarks tested Pearl's hierarchy on toy graphs with no data; data-science benchmarks tested code + analysis with no principled causal structure. CausalDS joins them by generating synthetic scenes with linked (SCM, data, narrative) triples, then evaluating LLM agents on jointly reasoning about the graph and analyzing the data.

## How it works

**Scene generation.** A generator samples a synthetic SCM, draws observational data from it, and wraps a domain-grounded narrative around it (e.g., "a clinical trial with these variables"). Optional observation noise adds realism.

**Task types.** Per instance, the benchmark asks tasks spanning:

- **Association** (Pearl rung 1) — "What is $P(Y \mid X)$?"
- **Intervention** (rung 2) — "What would $Y$ be if we set $X = x$?"
- **Counterfactual** (rung 3) — "Given this individual, what would $Y$ have been under a different $X$?"

Plus data-science operations (fit a model, compute an estimator) and tool use / code execution.

**Deterministic scoring.** Because the ground-truth SCM is known, every task has a computable correct answer.

**Non-identifiable cases are first-class.** Some queries (e.g., counterfactuals without sufficient data) cannot be answered from the observed distribution alone. CausalDS scores agents on whether they **abstain correctly** — declining rather than confabulating a number.

**Joint axes.** Rather than one aggregate score, the benchmark reports on five axes: symbolic causal reasoning, data science, uncertainty quantification, abstention, and tool use / coding. The dissociation between models is the payload.

## Why it matters

- **Names abstention as an axis.** Most LLM benchmarks penalize wrong answers but don't reward "I don't know." CausalDS treats correct abstention as first-class — closer to how a real data-science collaborator behaves.
- **Frontier vs. open-weight gap is epistemic.** The paper's headline finding — models dissociate primarily on epistemic axes, not symbolic reasoning — has implications for reward-model design. If open-weight models can reason but fabricate under ambiguity, the RL post-training reward has to price abstention correctly.
- **Ground truth via SCM.** Deterministic scoring on causal queries is a big deal — most causal-agent benchmarks rely on LLM-judge grading, which itself is noisy on causal questions. CausalDS sidesteps that.

## Gotchas & tricks

- **Synthetic-domain narratives.** The narratives are generated (not scraped from real datasets). Agents that heavily rely on domain memorization may look artificially weak; agents that reason from the data look strong. Read scores with the synthetic caveat in mind.
- **Non-identifiability is subtle.** Whether a query is identifiable depends on the graph and data. If your agent doesn't inspect the graph carefully, it will fabricate on non-identifiable queries; correct abstention is genuinely hard.
- **Tool-use coupling.** Because tasks require code execution to run estimators, sandboxing / tool-calling failures propagate. Separate tool-use scores from reasoning scores when comparing models.
- **Observation noise flips difficulty non-monotonically.** Small noise can make identifiable cases easier (as regularization); large noise flips them non-identifiable. Report noise settings when citing numbers.

## Sources

- Paper: *CausalDS: Benchmarking Causal Reasoning in Data-Science Agents* — Andrej Leban, Yuekai Sun (U. Michigan Statistics), 2026 — [arXiv 2607.08093](https://arxiv.org/abs/2607.08093).
- Reference: *Causality* — Pearl, 2009 — the three-rung causal hierarchy.
- Sibling benchmark: [uniclawbench](uniclawbench.md) — capability-decomposed agent eval.
