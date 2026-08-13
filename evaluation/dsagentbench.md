# DSAgentBench
*Depth — an agent benchmark grounded in its source paper.*

**TL;DR:** A benchmark of **275 end-to-end data-science tasks** run in *real computer environments* — notebooks, IDEs, terminals, browsers, and databases — covering the full data-science lifecycle (wrangling, exploration, modeling, visualization, validation). Uses **deterministic evaluators** that grade **analytical correctness, visual outputs, and model performance** rather than only code execution. Even the strongest agent (**Claude-4.6-Sonnet at 56.7%** task success) is far from saturated; all open-source agents stay **below 1%**. Introduced in *DSAgentBench* (York / NTU / Salesforce, 2026).

**Prereqs:** *(none)*
**Related:** [../agents/README.md](../agents/README.md), [humaneval.md](humaneval.md), [livecodebench.md](livecodebench.md)

---

## What it is

Data-science automation benchmarks have historically:

- Run in **sandboxes** with a script + oracle (limits realism).
- Evaluated **code correctness** rather than the *artifacts* a data scientist actually produces (charts, models, reports).
- Cover single steps (only wrangling, only modeling) rather than **end-to-end** workflows.

DSAgentBench swaps all three:

- The **environment is a real OS** with real tools — notebooks, IDEs, terminals, browsers, databases.
- The **grader inspects artifacts**, not just script exit codes.
- Tasks are **end-to-end** across the data-science lifecycle.

## How it works

**Task shape.** 275 tasks, each specifying a data-science problem with input data and an expected artifact set (a notebook cell output, a trained model file, a plot, a written summary). The agent runs in a real computer environment and can freely open a notebook, launch a terminal, browse for documentation, or query a database.

**Deterministic evaluators.** Instead of an LLM judge, each task ships evaluators that:
- Check analytical correctness against ground-truth statistics or a held-out sample.
- Inspect visual outputs (plot exists, axes labeled correctly, expected trend present).
- Measure model performance (accuracy / metric on a held-out split) against a task-defined threshold.

This is stricter than code-only execution grading — a script that runs without errors but produces the wrong plot fails.

**Environment realism.** The agent uses the same tools a human data scientist would. Latency, tool discovery, and file-system navigation cost are all in scope.

## Why it matters

- **Grounds the "AI data scientist" claim.** Product marketing outruns benchmark support constantly; DSAgentBench gives a concrete number to argue about.
- **Exposes the closed/open gap.** **Claude-4.6-Sonnet: 56.7%** vs **all open-source agents: <1%** is a striking gap. Some of that is the base model; a lot of it is the agent stack.
- **Artifact-level grading is the right level.** For real data-science automation, "the code ran" is not the metric that matters — the plot / model / report does.
- **Long-horizon computer-use signal.** DSAgentBench joins WebArena, SWE-bench, τ-bench, and VibeLifeBench as evidence that real-computer, long-horizon workflows are the current frontier for agents.

## Gotchas & tricks

- **Cost per task is high.** Real-environment agents run for many minutes each; benchmark sweeps are expensive.
- **Deterministic evaluators require careful design.** Small changes in a plot's axis or label can break naive checks — the evaluator library needs to be robust or it produces false negatives.
- **Open-source ceiling is fragile.** The <1% number will move rapidly as open agent stacks mature; treat it as a snapshot.
- **Environment leakage.** The benchmark's environment surface (available packages, file paths) can accidentally overlap with training data. Fresh open datasets help.

## Sources

- Paper: *DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments?* — Rahman, Islam, Mahbub, Laskar, Joty, Prince (York University / NTU / Salesforce AI Research), arXiv 2608.10366, 2026.
