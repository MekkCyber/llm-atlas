# Harness Evolution Evaluation Protocol
*Depth — matched-budget, held-out task protocol for evaluating automatic agent-harness search.*

**TL;DR:** Existing "automatic harness evolution" papers search harness configurations using task feedback, then evaluate on the *same* benchmark. Two flaws: (a) no budget-matched comparison against simple task-level search baselines, and (b) benchmark reuse leaks the search into the test. This paper's protocol fixes both — match feedback and inference budgets across methods, and hold out tasks — and shows automatic harness evolution then **does not consistently beat plain test-time scaling**.

**Prereqs:** [ifeval.md](ifeval.md)
**Related:** [../agents/README.md](../agents/README.md), [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

Automatic harness evolution treats an LLM agent's harness — its scaffolding, tool wiring, retry logic, verifiers — as a search space, and iteratively evaluates and revises candidate harnesses using unit-test feedback. Reported gains have been strong. The **evaluation is broken**: search uses the same benchmark as final reporting, and there's no comparison against just spending the search compute on more test-time samples.

This paper is the corrective protocol.

## How it works

### The two reforms

1. **Matched feedback + inference budget.** Compare harness evolution against **task-level search baselines** (best-of-N, self-consistency, verifier-guided search) that receive the same number of task feedback signals and the same number of inference tokens. This isolates whether gains come from *harness design* or from *more search compute at test time*.
2. **Held-out task set.** Evaluate discovered harnesses on tasks *not used during harness search*. If gains reflect real harness improvements, they should transfer; if they reflect overfitting to the search benchmark, they won't.

### The experimental setup

- **Testbed:** Terminal-Bench 2.1 — long-horizon terminal-manipulation agent tasks.
- **Models:** GPT-5.4 and Claude Opus 4.6 as the underlying agents.
- **Baselines:** Simple test-time scaling (multi-sample majority / best-of-N with a verifier), matched-budget.

### The findings

- Automatic harness evolution **does not consistently outperform** simple test-time scaling under matched feedback + inference budgets.
- Evolved harnesses show **limited generalization** to held-out tasks — a significant fraction of the gain evaporates.

The conclusion is not "harness evolution is fake" but "the reported gains are overstated because the protocol conflates search and evaluation."

## Why it matters

- **A field-level correction.** Almost every automatic-agent-scaffolding paper published in the last year uses the flawed protocol. Their headline numbers need to be recomputed under this one.
- **Portable protocol.** Matched-budget + held-out generalizes to any agent-optimization method (auto-prompting, auto-tool-selection, auto-verifier-design).
- **Reframes what "harness improvement" means.** If a harness doesn't beat matched-budget test-time scaling, its published gain is a search-budget effect, not a design effect.

## Gotchas & tricks

- **Matching "feedback budget" is not the same as matching "compute budget."** Feedback = task-level reward signals; compute = inference tokens. Both need matching, otherwise the comparison still favors whichever method has more of one.
- **Held-out tasks need to be genuinely disjoint.** Same-corpus, similar-difficulty held-outs still leak; use a distinct task family if possible.
- **The critique doesn't rule out harness evolution.** It rules out the *current* evaluation. Under a fair protocol, a harness-search method that finds structural improvements (not just more prompt tuning) can still win — the burden of proof shifts.
- **Applies to auto-eval too.** Any "let the LLM discover the right prompt/harness/verifier" scheme benefits from matched-budget + held-out. It's a cheap sanity check.

## Sources

- Paper: *Rethinking the Evaluation of Harness Evolution for Agents* — Wang, Zhu, Hu, Yuan, Chen, Senthil, Hajishirzi, Tsvetkov, Dasigi, Xiao — Allen Institute for AI / University of Washington, 2026.
