# Multi-LCB
*Depth — the multilingual extension of LiveCodeBench.*

**TL;DR:** A 12-language extension of [LiveCodeBench](livecodebench.md) that converts its **functional** and **STDIN/STDOUT** problems into a language-agnostic harness so each problem can be evaluated across Python, C++, Java, Rust, Go, TypeScript, JavaScript, C#, Kotlin, Swift, Ruby, PHP. Keeps LCB's release-date contamination filter. Evaluating 24 models, the paper shows large per-language disparities, clear **Python overfitting**, and **language-specific contamination** patterns invisible to the original Python-only LCB. Ivanova et al. (GigaCode + Yandex), 2026.

**Prereqs:** [livecodebench](livecodebench.md)
**Related:** [humaneval](humaneval.md) · [codeforces-benchmark](codeforces-benchmark.md) · [../data/decontamination.md](../data/decontamination.md)

---

## What it is

LCB problems come in two formats:

- **Functional** — implement a named function with typed inputs/outputs.
- **STDIN/STDOUT** — read from standard input, print to standard output (the competitive-programming style).

Multi-LCB ports both formats to a **language-agnostic execution harness**: per-language test runners that accept the same problem definition and call the language-specific entry point. The hidden-test sets are unchanged (these are platform tests scraped from LeetCode / AtCoder / Codeforces) — only the candidate-program side varies.

Each problem is tagged with the same **contest release date** as LCB, so contamination control is preserved per language. The reported numbers are **per-language Pass@1** on a release-date-filtered subset.

Covered languages (12): **Python, C++, Java, JavaScript, TypeScript, Go, Rust, C#, Kotlin, Swift, Ruby, PHP**.

---

## How it works

### Conversion pipeline

```
LCB problem (Python entry, hidden tests)
  → extract function signature / IO contract
  → render canonical problem prompt in target language idiom
  → wrap hidden tests in target-language test runner
  → run candidate code inside isolated sandbox per language
```

The **prompt template** is the load-bearing piece: each language gets idiomatic naming (snake_case vs camelCase vs PascalCase), idiomatic IO patterns (`println!` for Rust, `System.out.println` for Java), and idiomatic data types (`std::vector` vs `[]int`).

### Metric

Per-language **Pass@1** on the contamination-filtered subset. The paper also reports a **Python-relative gap** per model — i.e. how much each model drops from its Python score when forced into another language, isolating Python overfitting from genuine capability.

### Contamination handling

Same release-date filter as LCB ("after model cutoff date X"), but applied per **language**: a model with Python-dominated training data may have seen Python solutions for a 2024 problem published on a forum even before the contest date. The paper introduces a **language-specific contamination probe** by comparing pre-cutoff vs post-cutoff Pass@1 gaps per language.

---

## Why it matters

- Closes the most-cited blind spot in code-LLM evaluation: **the entire leaderboard has been Python-shaped**.
- Provides a clean operationalization of **language-specific contamination**, which prior benchmarks (HumanEval-X, MBPP-X) didn't measure.
- Likely to become the new default code benchmark for code-LLM training pipelines that care about non-Python deployment.

## Gotchas & tricks

- **Per-language test quality varies.** Languages whose problems were scraped from platforms with strict typed test runners (Rust, Java, C++) get cleaner pass/fail signals than dynamically-typed ones (Ruby, PHP).
- **Aggregate score is misleading.** A model strong on Python but weak on 11 others can match a uniformly-mediocre model on aggregate. **Always report per-language**.
- **Codeforces fraction is small.** Like LCB itself, Multi-LCB is dominated by LeetCode + AtCoder problems. For Codeforces-style evaluation use [codeforces-benchmark](codeforces-benchmark.md).
- **Prompt-template choice leaks information.** Including a function signature in the prompt biases toward signature-completion-style models; pure STDIN/STDOUT problems bias toward competitive-programming styles. Both modes are reported in the paper.
- **Cutoff dates differ per model.** Always pin the time window in any cross-model comparison.

## Sources

- Paper: *Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages* — Ivanova, Zadorozhny, Levichev, Petrov, Adamenko, Lopatin, Kutalev, Babaev (GigaCode + Yandex), 2026, arXiv 2606.20517.
- Paper: *LiveCodeBench: Holistic and Contamination Free Evaluation of LLMs for Code* — Jain et al., 2024, arXiv 2403.07974 — the Python-only original.
- Related: HumanEval-X (multilingual HumanEval) and MBPP-X — earlier multilingual code benchmarks without contamination filtering.
