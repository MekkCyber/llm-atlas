# Representation Axioms for LLM Latent Thoughts
*Depth — a benchmark-free, axiomatic protocol for evaluating hidden-state representations.*

**TL;DR:** Four functional axioms — **Causality, Minimality, Separability, Stability** — each with a quantitative metric computed directly on hidden states, independent of any downstream benchmark. The point is to separate "the representation is bad" from "the model on top of the representation is bad," which task accuracy alone conflates.

**Prereqs:** [interpretability README](../interpretability/README.md)
**Related:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md), [evaluation README](../evaluation/README.md)

---

## What it is

A meta-evaluation framework for what people loosely call "latent thoughts" — the internal representation of intermediate reasoning that LLMs are claimed to develop. Rather than measuring a task score and attributing the result to the representation, the framework probes the representation itself with four functional tests.

## How it works

For an LLM with hidden state `h(x)` at some chosen layer:

- **Causality.** Interventions on `h(x)` should change the output in predictable ways. Measured by counterfactual interventions and reading downstream effect on logits / next-token distribution.
- **Minimality.** `h(x)` should not encode information irrelevant to the task. Measured by how much task-irrelevant content can be linearly read off `h(x)` vs `x`.
- **Separability.** `h(x)` should distinguish *different questions* within the same task — not merely cluster by task type. Measured by pairwise classifier accuracy on intra-task question pairs.
- **Stability.** Equivalent inputs (paraphrases, formatting changes) should map to similar `h(x)`. Measured by representation distance between paraphrase pairs.

Each axiom yields a scalar score; no aggregation into a single number — failures on one axis are diagnostically more useful than a hidden average.

## Why it matters

- **Decouples representation quality from model capacity.** Two models can hit the same benchmark accuracy with very different representations; this framework exposes the difference.
- **A reproducible counterweight** to claims that long-CoT RL or distillation "improves reasoning representations." The audit found **no** open-weight model satisfying all four axioms across 23 reasoning tasks, and the deficits were **structural** (consistent across dense, distilled, RL-trained families).
- **Concrete diagnostic.** "Separability is failing" tells a researcher what to fix; "MMLU dropped 1 point" doesn't.

## Gotchas & tricks

- Choice of layer matters; the paper reports a sweep but a single layer's score should never be reported in isolation.
- Causality interventions are sensitive to how you parameterize the edit (linear projection vs activation patching) — pick one family and stick with it across compared models.
- The axioms are *necessary*, not sufficient. Passing all four does not imply the representation is "thinking"; it only rules out the documented failure modes.

## Sources

- Paper: *Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs* — Fahd Seddik, Fatemeh Fard, University of British Columbia — arXiv:2606.27378 — https://arxiv.org/abs/2606.27378
- Code: https://fard-lab.github.io/formalize-thoughts
