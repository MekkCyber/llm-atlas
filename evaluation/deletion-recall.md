# Deletion Recall (& Guard-and-Go)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A supplementary metric for LLM code-editing evaluations that measures whether the model *removes* the code it should — not just whether tests pass. Frontier models score 92%+ on right-file localisation but under 52% on cutting the exact line, and 29% of test-passing patches instead **wrap the offending code in a guard/fallback** (a pattern the authors call **Guard-and-Go**). Fills a blind spot in test-only metrics like SWE-bench resolve rate.

**Prereqs:** [swe-bench](swe-bench.md)
**Related:** [humaneval](humaneval.md), [livecodebench](livecodebench.md)

---

## What it is

Test-passing patches often "work" while making a codebase harder to maintain. Concretely: instead of deleting broken code, an LLM patch may wrap the old logic in an `if`/`try` guard, add a fallback path, or leave the original branch as dead code. Tests pass; the codebase gets messier.

**Deletion recall** measures the model patch's overlap with the developer patch's *deletions* specifically:

$$
\text{DelRecall} = \frac{|\text{lines removed by both model and developer}|}{|\text{lines removed by developer}|}
$$

**Guard-and-Go** counts patches that: (a) pass all tests, (b) retain code that the developer patch removed, and (c) add a guard/fallback around the retained code. It's a specific, detectable pattern rather than a fuzzy notion.

## How it works

**Computing DelRecall.**

1. Diff the developer patch — extract the set $D$ of deleted (file, line) pairs.
2. Diff the model patch — extract its deleted set $M$.
3. Line-level correspondence is done with a fuzzy match tolerating trivial whitespace differences.
4. Report $|M \cap D| / |D|$ per instance, averaged over instances.

**Detecting Guard-and-Go.**

1. Filter to patches that pass all tests.
2. For each such patch, check whether the developer-deleted lines are still present in the model's post-patch file.
3. If yes, check whether they now sit inside a new conditional / try-except / early-return added by the model.
4. That intersection is the Guard-and-Go set.

The paper releases this pipeline on top of SWE-bench Verified so any team can compute both metrics alongside resolve rate.

## Why it matters

- **Names a concrete failure mode** behind the widely-noted "LLM patches pass tests but degrade code quality" complaint.
- **Deletion recall is a strong verifiable reward** for RL-training coding agents — it penalises Guard-and-Go directly and is as easy to compute as resolve rate.
- **Sets an upper bound.** Even on tasks all top-5 SWE-bench Verified models solve, average deletion recall caps at 71.7%. There's headroom.
- Encourages a shift from "did the tests pass?" to "did the patch look like a developer would have written it?" without needing style-based human labels.

## Gotchas & tricks

- **Fuzzy-match granularity.** Whitespace-only or trivial-rename diffs shouldn't count as separate deletions — the paper's fuzzy match handles this. A naïve exact-string match tanks the metric.
- **Rewrites vs deletions.** If the model rewrites a function entirely (deletes the old version, adds a new one), the deletion set may include lines the developer patch also removed — count them as recalled.
- **DelRecall is not correctness.** A patch can achieve high DelRecall while breaking behaviour (deleting too aggressively). Use alongside resolve rate, not instead.
- **Guard-and-Go detection has false positives.** A legitimate patch may introduce a *new* conditional around *different* code; require the guard to actually wrap the developer-deleted lines.
- **Training on this metric risks reward hacking** — models could learn to delete conservatively to satisfy DelRecall while ignoring the task. Combine with resolve rate; scale weightings carefully.

## Sources

- Paper: *To Add Is Machine, To Delete Is Human: Measuring and Mitigating Deletion Avoidance in LLM Code Editing* — arXiv:2607.28887, 2026.
- Related: SWE-bench Verified — the substrate the paper's numbers come from.
