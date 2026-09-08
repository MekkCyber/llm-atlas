# Edit Fidelity for Code Repair
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Frontier code-editing models frequently rewrite code beyond the minimum needed to fix a bug. Edit fidelity — the ratio of the model's edit to the *known-minimal* patch — is a distinct quality axis from correctness (Pass@1) and can be measured, prompted, and *learned*. A benchmark built by injecting AST-level corruptions into BigCodeBench reference solutions gives each task a ground-truth minimal patch to compare against.

**Prereqs:** [humaneval.md](./humaneval.md), [livecodebench.md](./livecodebench.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

Existing code-repair benchmarks (HumanEval, LiveCodeBench, SWE-bench) measure whether the model's patch makes the test suite pass. They do not measure *how big* the patch is or *how much of the original is preserved*. In practice, over-editing wastes review effort, introduces regressions in untouched areas, and inflates git diffs to the point of un-reviewability.

Edit fidelity names the axis: given a known minimal patch, how close does the model's patch come to that minimum?

## How it works

- **Benchmark construction.** Start from 400 BigCodeBench problems. Inject controlled **AST-level corruptions** into reference solutions (e.g. swap an operator, delete a branch, rename a variable). Each corruption has a known, minimal reversing edit.
- **Metrics.**
  - **Excess Levenshtein distance** — how much longer the model's edit is than the minimal edit, in string-edit distance.
  - **Added cognitive complexity** — increase in complexity score (McCabe / cognitive) of the patched function beyond what the minimal fix requires.
  - **Pass@1** — the standard correctness axis, kept for cross-plotting.

- **Interventions studied.**
  - **Preservation prompt.** A system instruction to keep changes minimal. Reduces excess Levenshtein 0.195 → 0.131, cuts added cognitive complexity 26.6%, and *raises* Pass@1 by 2.3 pts.
  - **Bigger model / bigger reasoning budget.** Does *not* fix over-editing on its own.
  - **SFT on minimal patches.** Overfits to the *seen* corruption patterns; edit fidelity collapses out-of-distribution.
  - **RL on minimal patches.** Gives the best out-of-domain trade-off between edit fidelity and performance retention.

## Why it matters

- **Names a failure mode every user sees.** Coding-agent users routinely watch models rewrite unrelated helper functions. Edit fidelity turns that anecdote into a measurable regression axis.
- **Same benchmark surfaces the SFT-vs-RL divide.** SFT memorizes corruption patterns; RL generalizes. Clean example of the broader post-training story that RL generalizes off-distribution and SFT does not — see [../post-training/_rl.md](../post-training/_rl.md).
- **Cheap-to-produce data.** AST-level corruptions can be scaled to any code corpus, giving reliable minimal-patch ground truth without hand-labeling.

## Gotchas & tricks

- **Correctness ↔ fidelity is not zero-sum.** The preservation prompt raises *both* Pass@1 and fidelity — evidence that over-editing is often the *cause* of introduced regressions, not just visual noise.
- **AST corruptions are one distribution.** Real-world bug distributions include semantic bugs the AST-corruption harness does not sample; treat scores as directional.
- **Cognitive complexity has caveats.** McCabe / cognitive-complexity metrics reward structural minimalism even when a slightly larger refactor would be objectively better. Cross-check on Levenshtein.
- **RL signal quality.** RL rewards need a reliable minimal-patch oracle; noisy oracles teach the model to game whatever proxy you use (small diffs that fail tests, edits that pass tests but delete required behavior).

## Sources

- Paper: *When Models Edit Too Much: On the Fidelity of Minimal Code Edits* — Tongyao Zhu, Wei Hern Lim, Min-Yen Kan — NUS — arXiv:2609.04061 (2026).
- Benchmark: *BigCodeBench* — Zhuo et al., 2024 — the source corpus for the 400 problems.
