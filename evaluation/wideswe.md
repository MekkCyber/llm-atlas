# WideSWE
*Depth — a coding-agent benchmark for tasks that require coordinated changes across multiple repositories.*

**TL;DR:** WideSWE is 120 real cross-repo tasks (60 bug fixes, 60 features) mined from 103 software ecosystems, with hidden tests adapted to accept diverse correct implementations while preserving required behavior and regression checks. Full task success across seven agent configurations ranges 10.83%–42.50%; Codex CLI + GPT-5.6-sol tops out, exposing planning-across-contexts as the current agent bottleneck.

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** [livecodebench](livecodebench.md), [humaneval](humaneval.md), [codeforces-benchmark](codeforces-benchmark.md)

---

## What it is

A benchmark that stresses the coordination and cross-context planning abilities existing single-repo coding-agent benchmarks skip. Unlike SWE-bench (one repo per task), WideSWE tasks require the agent to identify and edit multiple repos coherently — for example, changing a library API and its accompanying framework in the same task.

## How it works

- **Mining.** Real ecosystems (SDK + service, library + docs, framework + plugin) were mined for issue/PR pairs whose fixes spanned multiple repositories. The task is derived from the linked issue and PR text.
- **Tests.** Existing hidden tests are systematically adapted to allow multiple correct implementations — the exact author solution is not required, only that behavior contracts and regression tests pass.
- **Split.** 120 tasks total, balanced 60/60 between bug fixes and features across 103 ecosystems.
- **Evaluation harness.** Two execution modes: *joint* (agent has simultaneous access to all repos) and *independent* (agent works one repo at a time with a shared plan). Seven concrete agent configurations tested (Codex CLI, OpenHands, others) with several backbone models.

## Why it matters

Modern software engineering is majority cross-repo: SDK changes ripple into services, framework changes ripple into ecosystems, library changes ripple into docs. Single-repo benchmarks (SWE-bench, SWE-bench Verified) silently exclude that entire class of work, so agents that saturate them can still fail on realistic production tickets. WideSWE isolates the *planning across contexts* dimension, which the paper's failure-mode analysis flags as the dominant source of failure: agents miss that a change is needed in another repo, or recognize it and leave the work unfinished, or edit the required repos but don't fully satisfy the request. This gives a training signal that SWE-bench alone doesn't.

## Gotchas & tricks

- **Joint vs independent execution has real tradeoffs.** Independent execution mainly recovers *omitted* work (the agent forgets a repo) but is worse at fixing *bad implementations*; joint execution can use cross-repo signals during verification. Configure the evaluation carefully.
- **Test-suite adaptation is imperfect.** Some tasks accept "correct" implementations that were not the human author's choice — this is intentional but occasionally lets suboptimal agents earn credit; use the pass rate alongside human review for headline claims.
- **Repository state.** WideSWE fixes ecosystem repos to specific commits; contaminating the agent with newer versions (via `pip install`, ambient docs) invalidates the score.
- **Benchmark saturation risk.** With 120 tasks and best-of-7 already at 42.50%, memorization of full solutions will eventually cap it; treat trajectories as the real signal.

## Sources

- Paper: WideSWE: Can Coding Agents Coordinate Changes Across Repositories? — Wang, Wang, Wu, Wu, Zhi, Yin, 2026 — [arXiv:2609.33382](https://arxiv.org/abs/2609.33382)
- Code: [github.com/ZJU-ACES-ISE/WideSWE](https://github.com/ZJU-ACES-ISE/WideSWE)
