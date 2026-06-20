# Multi-LCB
*Depth — LiveCodeBench extended to twelve programming languages.*

**TL;DR:** A direct multilingual extension of [LiveCodeBench](livecodebench.md): same contest-scraped, contamination-aware, rolling-release design, but the problem set is **translated and validated across twelve languages including Python**. Closes LCB's most-cited weakness — that frontier reasoning models trained mostly on Python code may not generalize to Go / Rust / JS / Java / TS / C++ / C#. Each language is graded with its own hidden-test suite, and aggregate scores expose cross-language transfer (or lack of it).

**Prereqs:** [livecodebench](livecodebench.md)
**Related:** [humaneval](humaneval.md), [codeforces-benchmark](codeforces-benchmark.md)

---

## What it is

LCB's pipeline (LeetCode / AtCoder / Codeforces scrape → date-stamped release → hidden tests → pass@1) re-implemented for 12 languages. Each problem appears in every supported language with language-specific test harnesses and authoritative hidden tests per language. Time-windowed evaluation works identically — restrict to problems released after the model's training cutoff and you get a contamination-aware multilingual code score.

## How it works

- **Problem set.** The Python source problems from LCB serve as the seed; statements are language-agnostic, solutions and tests are produced per language.
- **Per-language hidden tests.** Tests are authored in the target language (not auto-translated from Python) so they exercise language-idiomatic edge cases (integer overflow in Java, error handling in Go, ownership in Rust).
- **Aggregate score.** A simple mean across languages, or per-language pass@1 reported separately. Per-language breakdowns expose where a model's code generalization actually lies.
- **Same time-window machinery as LCB.** Reporting "Multi-LCB 2025-09 → 2026-03" works the same way.

The twelve languages cover the modern production landscape: Python, JavaScript, TypeScript, Java, C++, C#, Go, Rust, Kotlin, Swift, PHP, Ruby (representative list — exact mix per the paper's table).

## Why it matters

- LCB is the *de facto* code-reasoning benchmark cited in every recent reasoning-model release, and its Python-only scope has been an open weakness. Multi-LCB is positioned to become the standard multilingual code eval in 2026.
- Exposes a model's **language transfer**: a model that scores 70% Python but 25% Rust tells you something specific about its training data mix and reasoning generality that pure-Python LCB cannot.
- For real software engineering deployment, Python-only correctness is rarely sufficient. This is the evaluation that matches the deployment surface.

## Gotchas & tricks

- Translation quality matters. If problem statements are paraphrased per language, scores aren't comparable across languages — check the methodology to confirm statements are language-invariant.
- Per-language test rigor varies (Python tests are mature; some other-language tests are likely thinner at launch). Treat per-language rankings cautiously until multiple model releases stabilize the leaderboard.
- Same contamination caveat as LCB: rolling release is *relative* to training cutoff, not absolute. Always report the time window used.
- Aggregate scores hide imbalance: a model that's strong in 11 languages and weak in 1 averages similar to one that's mediocre across the board. Show per-language tables.
- Composes with the four LCB scenarios (generation / repair / execution / test-prediction) — multilingual repair / execution evaluations are likely follow-ups.

## Sources

- Paper: *Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages* — Ivanova et al., GigaCode / Yandex SDA, 2026 — arXiv 2606.20517.
- Predecessor: *LiveCodeBench* — Jain et al., UC Berkeley + MIT + Cornell, 2024 — arXiv 2403.07974 (see [livecodebench](livecodebench.md)).
- Site: livecodebench.github.io (multilingual extension likely linked from the LCB site post-release).
