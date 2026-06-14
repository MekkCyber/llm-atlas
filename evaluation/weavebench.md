# WeaveBench
*Depth — long-horizon benchmark for computer-use agents that must mix GUI, CLI/code, browser, and external tools.*

**TL;DR:** Production computer-use agents bounce between visual desktop control, command-line execution, browsers, and external tools within one episode. Most benchmarks pin agents to one interface (GUI-only or CLI-only) and score final answers. WeaveBench scripts **114 tasks across 8 domains** that *require* hybrid interfaces and grades with **trajectory-aware judging**, so agents that brute-force a final state without doing the actual workflow are penalized. Best agents top out at **35.1% PassRate** on fixed OpenClaw, **41.2%** averaged across harnesses.

**Prereqs:** *(none)*
**Related:** [README.md](README.md) · [../agents/README.md](../agents/README.md)

---

## What it is

A benchmark of long-horizon tasks deployed in real runtimes (OpenClaw and others). Each task specifies a goal that *requires* coordination across at least two of: GUI control, CLI/code execution, browser navigation, external tool calls. Tasks are authored in 8 domains spanning office software, system administration, data wrangling, and web research.

The judging model has access to the agent's full trajectory (not just the final state) and scores against a rubric of expected actions plus a final-state check.

---

## How it works

### Runtimes

- **Fixed harness (OpenClaw)** — single deployed environment, controlled tool palette. Used for the headline number.
- **Multi-harness average** — same tasks run across multiple production harnesses (different OS variants, tool sets). Captures harness sensitivity.

### Scoring

Two-part rubric:
- **Final state check** — did the user-visible goal get achieved? (binary or scalar via simulator).
- **Trajectory check** — did the agent's intermediate steps match the workflow rubric? Penalizes agents that, e.g., manually edit a result file instead of running the requested command.

The combined score is the **PassRate**. Trajectory-aware judging is the main methodological contribution.

### Baselines

Frontier closed and open models reported on the leaderboard; the best score on fixed OpenClaw is 35.1%, multi-harness average 41.2%. Large gap between same-model scores across harnesses signals brittle generalization.

---

## Why it matters

- **Hybrid interface is the production setting**, not a research curiosity. Benchmarks that fix one interface mis-rank agents for real deployment.
- **Trajectory-aware judging closes a known gaming surface.** Agents that hallucinate the final state past brittle GUI steps are caught.
- **Cross-harness variance is now visible.** Treating "computer-use ability" as a single scalar hides harness-specific overfitting; WeaveBench surfaces it.

---

## Gotchas & tricks

- **Trajectory rubrics are author judgment.** Different rubrics weight "do it this way" vs "any path to the goal" differently — read the per-task rubric before drawing conclusions.
- **Determinism in the runtime matters.** GUI agents are sensitive to small environment differences (cursor positions, animation delays); reproducibility requires pinning the runtime image.
- **Long-horizon = long context.** Tasks routinely produce 10K+ token trajectories; agents with short context windows are at a structural disadvantage.
- **Doesn't cover production multi-tenant complications.** Concurrent users, real network failures, and long-lived sessions aren't represented.

---

## Sources

- Paper: *WeaveBench: A Long-Horizon, Real-World Benchmark for Computer-Use Agents with Hybrid Interfaces* — Li et al., MSRA, 2026 — [arXiv:2606.09426](https://arxiv.org/abs/2606.09426).
