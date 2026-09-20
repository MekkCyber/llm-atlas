# Coding-Agent Harness
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **coding-agent harness** is the scaffolding around a base LLM that turns single completions into long-horizon software-engineering trajectories: it defines the action space (tools, bash, editors), the planning layer, the context-management strategy, and the outer execution loop. Most reported "model wins" on SWE-Bench / Terminal-Bench hide harness wins — component-level ablations reveal that context management, planning depth, and action-space design each contribute independently and interact strongly with model strength and context-window budget.

**Prereqs:** [README.md](README.md)
**Related:** [../evaluation/livecodebench.md](../evaluation/livecodebench.md)

---

## What it is

A harness is everything between "call the model" and "the task is done":

- **Action space** — predefined tool schemas (edit_file, run_tests) versus a bash-only interface where the model composes commands itself.
- **Planning layer** — an explicit plan-then-execute step, sometimes a separate planner model.
- **Context management** — how the harness compresses trajectory history to fit the model's window (rule-based elision, LLM-based summarization, mixed).
- **Execution loop** — the outer control flow: how tool outputs feed back, how errors are handled, how termination is decided.

Fixing the outer loop and varying these components produces very different pass rates on the same base model, at the same or lower token cost.

## How it works

Two representative recipes:

- **Component ablation ("Empirical Study of Harness Design", 2026)** — 176 matched settings across 4 models × 5 context-management strategies × 4 window budgets on SWE-Bench Verified and Terminal-Bench 2.1. Four findings: (1) context management value scales with window pressure, mostly by avoiding overflow; (2) rule-based elision *before* LLM summarization beats every other context strategy; recoverable elision adds complexity models don't use; (3) planning is an accuracy scaffold for weak models and a cost saver for strong ones; (4) predefined tools help models with weak bash proficiency; bash-only wins for bash-capable models on CLI-heavy tasks.
- **RSI-inspired auto-search ("SoL-Pi", 2026)** — a search loop that proposes harness edits, evaluates across many rollout environments, and keeps only edits whose gains transfer. The four surviving mechanisms (action execution, context compaction, observation handling, delegated reading) hold performance at Pi levels while cutting token traffic 44.7–49.0% and API cost ~1/3.

## Why it matters

- Harness design is currently underspecified in most benchmark reports, which makes cross-model comparisons noisy.
- Serving cost lives here: context compaction and action-space choice together can cut token traffic by half without accuracy loss.
- Harness edits generalize across base models when found by transfer-aware search — a much smaller lift than model retraining for the same gains.

## Gotchas & tricks

- Recoverable-elision context strategies (keep summaries with pointers back to the full trajectory) look attractive but models rarely use the recovery path — you're paying for machinery that rounds to no-op.
- The right action space is *model-dependent*: forcing bash-only on a bash-weak model tanks pass rate; predefining tools for a strong bash model just adds latency.
- Always report the harness alongside the model — a "Pass@1 = X on Y" number is otherwise meaningless.

## Sources

- "An Empirical Study of Harness Design for Coding Agents" — Fan et al., 2026 — [arXiv:2609.20804](https://arxiv.org/abs/2609.20804).
- "SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness" — Liu et al., 2026 — [arXiv:2609.20519](https://arxiv.org/abs/2609.20519).
