# Multi-LCB
*Depth — LiveCodeBench, extended to 12 programming languages while preserving contamination-by-cutoff.*

**TL;DR:** Multi-LCB ports every Python LiveCodeBench problem to 11 additional languages (C++, Java, Go, Rust, JavaScript, Kotlin, PHP, Ruby, C#, etc.), keeping the rolling date-tagged release that makes LCB contamination-resistant. The headline finding from a 24-model sweep: per-language pass@1 gaps of 20–40 pp between Python and rarer languages, plus visible language-specific contamination spikes for models post-trained on a particular language. Maria Ivanova et al., GigaCode + Yandex, arXiv 2606.20517.

**Prereqs:** [livecodebench](livecodebench.md)
**Related:** [humaneval](humaneval.md) · [codeforces-benchmark](codeforces-benchmark.md) · [../data/decontamination.md](../data/decontamination.md)

---

## What it is

LiveCodeBench is Python-only. Multi-LCB translates the **problem statements + example I/O** (not the solutions) into 11 additional languages and regenerates hidden test harnesses per language, so the same conceptual problem can be scored in each. The contamination-control story carries over: each problem keeps its original LCB release-date tag, so any time-windowed report (e.g. "Multi-LCB 2025-06 → 2026-04") evaluates only on post-cutoff problems in every language.

12 languages in the v1 release: Python, C++, Java, Go, Rust, JavaScript, TypeScript, Kotlin, PHP, Ruby, C#, Swift (exact list per the paper).

## How it works

- **Problem translation.** Statements are translated automatically with quality checks; example I/O is preserved verbatim.
- **Per-language test harness.** A language-specific scaffold runs the model's solution against the hidden test set in a sandbox.
- **Metric.** Pass@1 per (model, language, problem), reported aggregated per language and per platform (LeetCode / AtCoder / Codeforces tier).
- **Time-window subset.** Same convention as LCB — report numbers on a window post-dating the model's training cutoff.

Cross-language consistency = std of per-language pass@1 across the 12 languages; low std signals true code-reasoning rather than Python pattern-completion.

## Why it matters

- **Multilingual code is undertested.** Pre-Multi-LCB, multilingual code eval relied on HumanEval-X and MultiPL-E, both of which have been in pretraining corpora for years.
- **Exposes Python-overfit models.** A 24-model sweep shows frontier models with strong Python LCB scores can drop 20–40 pp on lower-resource targets — invisible in Python-only reporting.
- **Catches language-specific contamination.** Spikes on a single language (typically the language a model was instruction-tuned on) become a measurable diagnostic.

## Gotchas & tricks

- **Translation quality is a confound.** Statement-translation errors can make a problem unsolvable in some languages; the paper screens for this but no screen is perfect.
- **Per-language hidden tests vary.** Some languages have fewer scraped contest tests than Python; LLM-generated tests can be weaker.
- **Cross-language consistency != true generality.** A model that's strong in two languages and weak in 10 can still have a respectable mean. Always inspect per-language tables.
- **Window discipline still required.** Multi-LCB inherits LCB's "Live" property only if you report on a post-cutoff window.

## Sources

- Paper: *Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages* — Ivanova, Zadorozhny, Levichev, Petrov, Adamenko, Lopatin, Kutalev, Babaev, GigaCode + Yandex School of Data Analysis + Applied AI Institute, 2026, arXiv 2606.20517.
- Paper: *LiveCodeBench* — Jain et al., 2024, arXiv 2403.07974 — the original benchmark Multi-LCB extends.
