# SWE-bench Science
*Depth — a SWE-bench-style benchmark where "correct" means the fix preserves the scientific evidence the code produces, not just green CI.*

**TL;DR:** 119 code-repair tasks drawn from real scientific software across 20 domains (chemistry, astronomy, bioinformatics, physics simulation, …). Unlike vanilla SWE-bench, correctness includes *numerical faithfulness of scientific outputs*, not only test-suite pass. Frontier coding agents (Claude Code + Opus-5) score **< 50% pass@1**, with four recurring failure taxonomies: deficits in scientific knowledge, misguided exploration, incomplete repair coverage, and overfitting to observed cases.

**Prereqs:** [README.md](README.md), [livecodebench.md](livecodebench.md)
**Related:** [../agents/README.md](../agents/README.md), [../agents/executable-task-synthesis.md](../agents/executable-task-synthesis.md), [humaneval.md](humaneval.md)

---

## What it is

A benchmark that operationalizes "would you let this agent touch code that produces published results?" Each task presents:

- A scientific-software repository (Python-heavy — SciPy, AstroPy, biology stacks, etc.).
- An issue description with the observed misbehavior.
- A hidden ground-truth patch.
- A verifier that both **runs the repo's tests** and **checks numerical faithfulness** of key outputs against reference values.

Scored as pass@1 over the 119 tasks. Publicly available: [github.com/OpenMOSS/SWE-bench-Science](https://github.com/OpenMOSS/SWE-bench-Science), dataset on HuggingFace.

## How it works

The scoring pipeline for each task:

1. Agent gets the repo + issue text and produces a patch.
2. Verifier applies the patch, runs the test suite.
3. Verifier also runs designated *scientific outputs* (numerical simulations, statistical estimates) and checks tolerances against ground truth.
4. Task passes only if both (2) and (3) succeed.

Failure mode categorization (from the paper's analysis of ~600 failed trajectories):

| Category | What goes wrong |
| --- | --- |
| **Scientific knowledge deficit** | Agent doesn't know the underlying physics/math/algorithm — patches the symptom, breaks the semantics. |
| **Misguided exploration** | Agent reads the wrong files, tests the wrong hypotheses; spends the budget in a dead branch. |
| **Incomplete repair coverage** | Fix works for the reported case, misses adjacent code paths that share the bug. |
| **Failure to generalize beyond observed cases** | Overfits to the specific inputs mentioned in the issue; regresses on other inputs. |

Providing extra scientific context (papers, method summaries, formula references) has *mixed* effects: on some tasks it helps; on others it derails exploration by anchoring the agent on the wrong direction.

## Why it matters

Existing SWE benchmarks measure "does my library still work?" This one measures "would the output of this repo remain scientifically defensible?" — a critical property for domains (science, medicine, finance, systems verification) where correctness is more than green CI. The four failure taxonomies also form a diagnostic map for agent-improvement work: each category maps to a distinct capability to target.

## Gotchas & tricks

- **Contamination-resistant by construction.** Tasks are drawn from active scientific-code issues; harder to memorize than curated benchmark instances. Still, monitor training data.
- **Numerical tolerances are the sharp edge.** A patch that changes a random seed, floating-point path, or ordering can silently break scientific-output checks that pass structural tests. Design agent tools to preserve determinism where possible.
- **Guidance helps unevenly.** Injecting scientific context works for tasks bottlenecked on knowledge; it hurts on tasks bottlenecked on code exploration. Consider gating on failure-mode diagnosis before adding it.
- **Not a pure coding benchmark.** Agents strong on SWE-bench Verified but weak on scientific literacy underperform here. The rankings genuinely differ.

## Sources

- Paper: *SWE-bench Science: Can Coding Agents Resolve Engineering Tasks in Science?* — Xu, Lu, Zheng, Wang, Qiu, OpenMOSS/Fudan, 2026 — [arXiv:2608.19799](https://arxiv.org/abs/2608.19799).
- Code: <https://github.com/OpenMOSS/SWE-bench-Science>
- Data: <https://huggingface.co/datasets/OpenMOSS-Team/SWE-bench-Science>
- Project page: <https://swescience.github.io>
