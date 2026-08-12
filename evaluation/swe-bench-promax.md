# SWE-Bench ProMax
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A 170-instance, seven-language (Python, Java, TypeScript, Go, C, C++, Rust) benchmark for **large-scale code refactoring**, engineered as the successor to SWE-bench Verified. Each instance averages 11.4 modified files and 261.6 LOC. Issue specs are rewritten from scratch and tests are manually audited to remove the "narrow tests reject correct patches" and "broad tests check unstated requirements" failure modes identified in prior benchmarks. Best current model resolves 41.2% — unsaturated and much harder than SWE-bench Verified.

**Prereqs:** [swe-bench.md](swe-bench.md)
**Related:** [livecodebench.md](livecodebench.md), [humaneval.md](humaneval.md)

---

## What it is

SWE-bench Verified was audited in 2025 and found to have two systematic issues: ~60% of "unsolved" instances contained flawed tests (either overly narrow tests that reject correct solutions, or overly broad tests checking unstated requirements), and frontier models could reproduce gold patches from training data. SWE-Bench ProMax rebuilds around three fixes: move from bug-fixes to **refactors**, expand to **multiple languages**, and **rewrite specs / audit tests** from scratch.

## What's different from SWE-bench Verified

| Dimension | SWE-bench Verified | SWE-Bench ProMax |
|---|---|---|
| **Task type** | Single-issue bug fixes | Behavior-preserving refactors |
| **Languages** | Python only | Python, Java, TypeScript, Go, C, C++, Rust |
| **Instance count** | 500 | 170 |
| **Files per instance (avg)** | 1–3 | **11.4** |
| **LOC per instance (avg)** | ~15 | **261.6** |
| **Spec source** | Original GitHub issue | Rewritten from scratch |
| **Test audit** | OpenAI manual review | Multi-stage manual review, dropping narrow/broad tests |
| **Best model resolve rate** | 60%+ (saturated) | **41.2%** (unsaturated) |

Refactoring stresses long-horizon planning and cross-file consistency in a way single-issue bug fixes don't. There's also less contamination risk — refactor commits are less well-indexed than issue-driven fix commits.

## Why it matters

- **Replaces a saturated benchmark.** With frontier models nearing 60% on Verified and gold-patch memorization documented, Verified is no longer a good discriminator. ProMax's 41.2% best resolve rate provides headroom.
- **Multilingual makes single-language contamination less useful.** A Python-only training-data cheat doesn't help on the Go / Rust instances.
- **Scale-realistic.** 11.4-file changes at 261.6 LOC are closer to real engineering work than 1-file 15-LOC fixes. Agents that succeed here have to actually plan across a repo.
- **Audit process is explicit.** The paper documents the multi-stage curation (spec rewrite, narrow-test removal, broad-test removal) — a template other benchmark authors can copy.

## Gotchas & tricks

- **"Refactoring" is a fuzzy concept.** ProMax includes rename-across-file, extract-method, cross-module dependency inversion, and API surface reshaping. Different scaffolds may specialize on different subtypes.
- **Test coverage sets the ceiling.** Even with the audit, refactoring correctness is inherently under-tested by any finite test suite — a patch can pass tests but subtly change semantics.
- **Scaffold choice matters more than on Verified.** The 41.2% number was reported under two agent scaffolds; the delta between scaffolds is substantial. Report scaffolds explicitly.
- **Language mix has different difficulty profiles.** C / C++ instances are harder on average (build complexity, memory safety); TypeScript / Python are easier. Aggregate resolve rate hides this.
- **170 instances is small.** Confidence intervals on aggregate resolve rate are wider than on 500-instance Verified — differences under ~5 pts are noise.

## Sources

- Paper: *SWE-Bench ProMax: Benchmarking Agents on Large-Scale Multilingual Code Refactoring* — 2026 — the source paper.
- Related: [swe-bench.md](swe-bench.md) for the original benchmark family this succeeds.
- Related: audit of SWE-bench Verified (2025) motivating the redesign.
