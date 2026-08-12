# SWE-bench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark of real-world GitHub Python issues plus their gold-patch fixes, evaluated by running the repo's own test suite against the model's proposed patch. Introduced 2023 as an ambitious agent-capability yardstick, saturated by 2024–2025, then re-cast in successive variants (Verified, Multimodal, Lite, ProMax) to keep it useful.

**Prereqs:** [livecodebench.md](livecodebench.md), [humaneval.md](humaneval.md)
**Related:** [swe-bench-promax.md](swe-bench-promax.md)

---

## What it is

Each SWE-bench instance is a real closed GitHub issue plus the commit that fixed it. The model receives: the issue text, the repo at the pre-fix commit, and the repo's test infrastructure. The model must produce a patch that (a) applies cleanly and (b) passes the tests that the human's gold patch was designed to make pass — including tests the model *cannot see*.

Original SWE-bench: 2,294 Python issues drawn from 12 popular libraries (Django, Flask, scikit-learn, etc.). Grading is fully automatic — run the tests, look at pass/fail.

## Variants

- **SWE-bench Full.** All 2,294 instances. Broad but noisy — some instances have flawed tests or ambiguous issue descriptions.
- **SWE-bench Verified** (2024). 500 instances manually audited by OpenAI to remove flawed tests / underspecified issues. Became the de-facto agent-eval standard through 2025.
- **SWE-bench Lite.** 300-instance subset for cheap evaluation.
- **SWE-bench Multimodal.** JavaScript / TS variant with UI screenshots — extends beyond Python.
- **SWE-bench ProMax** (2026). 170 instances across 7 languages focused on **large-scale refactoring** rather than single-issue bug-fixes. See [swe-bench-promax.md](swe-bench-promax.md).

## Why it matters

- **First "real software" benchmark for agents.** Prior code benchmarks (HumanEval, MBPP) tested function-body synthesis in isolation. SWE-bench put models into real repos with real test suites.
- **Drove the coding-agent frontier.** Went from ~2% resolve rate at launch (2023) to 60%+ on Verified by early 2025 across frontier models plus custom scaffolds — a headline agent-capability trajectory.
- **Testbed for scaffolds.** The gap between a raw model and the same model inside a Devin / Aider / SWE-agent scaffold is often 20+ pts of resolve rate. SWE-bench made the "model × harness" product visible.
- **Contamination pressure prompted successors.** 2025 audits found frontier models could reproduce gold patches verbatim from training data. This is what triggers the ProMax redesign around refactoring rather than fixing.

## Gotchas & tricks

- **Test suite quality is uneven on Full.** Verified is the safer choice for reporting numbers.
- **Gold-patch memorization is a real threat.** Newer models trained on post-2023 web data may have seen the fixes. Numbers on Full/Verified should be interpreted with that caveat.
- **Resolve rate ≠ correctness.** A patch that passes the model-visible tests can still be functionally wrong (fails hidden tests, or breaks in production). The pass/fail metric is the ceiling on measurement fidelity.
- **Instance runtime varies wildly.** Some instances take seconds to grade; some require minutes of test suite runtime. Budget accordingly for full-suite evaluations.
- **Scaffold matters as much as model.** Reported numbers should always specify the scaffold — "GPT-5 on SWE-bench Verified" without the scaffold is under-specified.

## Sources

- Paper: *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* — Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan, 2023.
- OpenAI post: *Introducing SWE-bench Verified* — 2024 — the manual-audit subset.
- Paper: *SWE-Bench ProMax* — 2026 — refactor-focused multilingual successor. See [swe-bench-promax.md](swe-bench-promax.md).
