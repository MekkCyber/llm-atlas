# SchrödingerRepo
*Depth — functionality-preserving repository transformation for contamination-robust code benchmarks.*

**TL;DR:** Repository-level coding benchmarks (SWE-bench and its cousins) are built from popular open-source repositories that leak into pretraining. A static benchmark cannot separate reasoning from memorization. SchrödingerRepo applies four **functionality-preserving transformations** to each benchmark repository — problem-statement reconstruction, namespace remapping, intra-file layout reordering, and semantics-preserving code rewriting — so a model's performance drop must come from lost memorization rather than lost reasoning capability. Provides a refreshable evaluation surface for repository-level coding agents.

**Prereqs:** [decontamination](../data/decontamination.md), [livecodebench](livecodebench.md)
**Related:** [humaneval](humaneval.md)

---

## What it is

An evaluation-transformation protocol. Take an existing repository-level coding benchmark (e.g. SWE-bench) and produce an alternate copy of each repository that has the **same fix required** to solve the task but different surface memorization signals. The four transformations are stacked; each targets a distinct memorization channel.

## How it works

Four levels of transformation, applied to a benchmark's underlying repositories:

1. **Problem statement reconstruction.** The natural-language issue is rewritten (paraphrased, restructured, and de-jargoned) so that verbatim recall of the original issue text is useless.
2. **Namespace remapping.** Module, class, function, and variable names are renamed consistently across the repository — the code executes identically but familiar symbols (`class UserAuth`, `def parse_request`) are replaced with fresh ones.
3. **Intra-file layout reordering.** The physical order of independent definitions inside each file is permuted. Import blocks are reformatted, decorators are shuffled where legal, and top-level statements are moved without changing execution.
4. **Functionality-preserving code rewriting.** Semantics-preserving refactorings — loop-to-comprehension, comprehension-to-loop, control-flow flattening, expression reordering — remove stylistic fingerprints of the canonical repo.

Each level is applied and validated (unit tests still pass, the same fix location still resolves the bug). The final "Schrödinger" repository is functionally equivalent but visually different from what pretraining saw.

## Why it matters

The community treats SWE-bench performance as a comparable, stable capability metric. In practice, a substantial fraction of a model's SWE-bench score is repository memorization — the model recognizes the codebase from pretraining and knows what fix belongs to what issue. SchrödingerRepo separates the two components: models that lose few points under transformation are actually reasoning; models that collapse were mostly recognizing.

More broadly, this generalizes the LiveCodeBench "refresh over time" insight to full repositories, where refreshing by adding new tasks is expensive. Transformation is cheap.

## Gotchas & tricks

- Transformations have to be validated automatically — a subtle rewriting bug can invalidate the fix location and produce false negatives. The paper reports each level was individually stress-tested with the benchmark's own test suite.
- Some transformations are stronger against some models: namespace remapping mainly hurts models that leaned on symbol memorization; layout reordering hurts token-window-position memorization. Report per-level ablations, not just the stacked score.
- The technique doesn't fix contamination that comes from **behavior** memorization (recall of a specific bug pattern independent of surface form). That is a harder problem.

## Sources

- Paper: *Schrödinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It?* — Chen et al., 2026 — [arXiv:2609.27891](https://arxiv.org/abs/2609.27891)
