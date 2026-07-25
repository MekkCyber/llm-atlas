# Tencent WorkBuddy Bench
*Depth — a multi-domain coding-agent benchmark whose contamination resistance rests on task-construction obfuscation rather than dataset secrecy.*

**TL;DR:** Closed-set eval leaks. Tencent WorkBuddy Bench takes the opposite approach: every task is *open-sourced end-to-end* (task directory, environment image, tests, reference solution) but *not recoverable by web-searching the underlying commit / PR / issue*, because each prompt is rewritten as a short, colloquial, role-played request that shares no distinctive strings with its source. Four subsets probe repo-level engineering, front-end, office/business workflows, and red-/blue-team security, all packaged in a uniform task-directory format and run on both CodeBuddy Code and Claude Code harnesses.

**Prereqs:** [../evaluation/README](README.md), [../data/decontamination](../data/decontamination.md)
**Related:** [livecodebench](livecodebench.md), [humaneval](humaneval.md), [codeforces-benchmark](codeforces-benchmark.md)

---

## What it is

A publicly-released evaluation suite for coding *agents* (not just code-generation models). Each task ships as a directory with:

- environment image (Docker or similar) reproducing the target system,
- reference solution and tests,
- a colloquial user prompt reverse-engineered from a real commit / PR / business scenario,
- a scoring script appropriate to the subset.

Four subsets, four scoring instruments:

- **Repo-level engineering** — SWE-Bench-style multi-file patches, scored by test pass.
- **Front-end development** — visual + functional criteria, scored by a specialised harness.
- **Office / business workflows** — end-to-end task completion in office-style environments.
- **Red-/blue-team security** — offensive/defensive tasks with security-appropriate scoring.

Because the scoring instruments differ, no suite-wide average is reported — meaningful comparison lives *inside* each subset.

## How it works

**Task construction is the contamination story.** Every task follows a three-step recipe:

1. **Ground.** Pick a real commit, pull request, or business scenario (internal or public).
2. **Reverse-engineer.** Reconstruct the environment (repo state, files, dependencies) at the moment of the change.
3. **Rewrite.** Produce a short, colloquial, role-played user request that *would motivate* the underlying change without quoting or paraphrasing the original issue/PR text. The prompt shares no distinctive strings with any web-indexable trace of the change.

The full dataset is then released openly, so third parties can re-run every task and inspect it. Retention against contamination comes from **surface-form distance** at construction time plus **dataset versioning**, not from secrecy. This inverts the closed-eval assumption: hiding a test set delays contamination; obfuscation at construction time avoids it structurally.

The benchmark is run under a uniform, reproducible protocol on two agent harnesses (CodeBuddy Code and Claude Code) to separate model capability from harness confounds.

## Why it matters

- **Contamination-by-construction is portable.** The recipe (ground → reverse-engineer → rewrite) applies to any domain with real work artefacts.
- **First-class multi-harness runs.** Agent evaluations have historically been harness-locked; running the same tasks on two harnesses exposes how much of a score is the harness.
- **Realistic scope.** The four subsets cover the actual shape of enterprise coding work — engineering, front-end, business workflows, security — rather than the algorithmic slice most benchmarks stop at.

## Gotchas & tricks

- **No cross-subset average.** Scoring instruments differ enough that a single number would be misleading. Report per-subset scores.
- **Prompt-rewriting quality is the contamination ceiling.** A lazy rewrite that keeps distinctive strings re-opens the leak.
- **Harness comparison isn't apples-to-apples.** CodeBuddy Code and Claude Code have different tool sets and prompting conventions; the shared task directory helps, but harness-specific tuning still matters.
- **Versioned releases are load-bearing.** Contamination resistance depends on the dataset being versioned so old runs can be re-scored against the version they were run on.

## Sources

- Paper: *Tencent WorkBuddy Bench: A Multi-Domain Coding-Agent Benchmark with Contamination-Resistant Task Construction* — Cai et al. (Tencent Youtu / Keen Security / Workbuddy / Yunding Security), 2026 — [arXiv:2607.20911](https://arxiv.org/abs/2607.20911).
