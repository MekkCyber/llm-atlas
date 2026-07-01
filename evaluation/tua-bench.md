# TUA-Bench
*Depth — one specific benchmark, grounded in its source paper(s).*

**TL;DR:** A general-purpose benchmark for **Terminal-Use Agents** (TUAs): 120 real-world tasks across five task families — document editing, email, live-web info-seeking, scientific workflows, engineering workflows — each running in a real terminal with a deterministic setup script and execution-based grading. Fills the gap between GUI benchmarks (OSWorld family) and shell-coding benchmarks (SWE-bench, TerminalBench).

**Prereqs:** [README.md](./README.md), [../agents/README.md](../agents/README.md)
**Related:** [osworld2.md](./osworld2.md), [swe-together.md](./swe-together.md), [livecodebench.md](./livecodebench.md)

---

## What it is

Terminals are the "invisible OS" for professional work — document conversion, email triage, live-web scraping, HPC job control, scientific tooling. Existing agent benchmarks target either graphical desktops or programming-centric shell tasks; neither exercises general terminal use. TUA-Bench isolates that surface.

## How it works

**Task pool.** 120 tasks across five families:
- Document editing (transforms, format conversion, batch operations).
- Email management (search, triage, reply drafting from CLI).
- Live-web info-seeking (curl/wget/API composition to answer factual queries).
- Scientific workflows (co-designed with PhD-level domain experts; often specialised CLI tools).
- Engineering workflows (build systems, HPC schedulers, container orchestration).

**Reproducibility.** Each task ships a deterministic setup script — same starting state on every run. The agent operates in a real shell (not a simulator), so tool versions and side-effects are genuine.

**Grading.** Execution-based: a task-specific script inspects the terminal state / produced artifacts after the agent's turn ends. Binary success plus a partial-completion axis for multi-goal tasks.

**Leaderboard signal.** The paper reports Claude Code with Claude Opus 4.8 max reasoning effort at **65.8% overall** — the strongest frontier agent — with substantial gaps between the "routine digital activities" and "scientific/engineering" tracks.

## Why it matters

- **Closes an eval gap.** The terminal is the dominant surface for professional agent deployment (Claude Code, Codex, Cursor CLI mode) and lacked a shared, non-coding-centric benchmark.
- **Execution-based grading is hard to game.** Unlike LLM-judged benchmarks, task completion is verified by inspecting real filesystem/process state.
- **Cross-track scoring identifies specialisation gaps.** Strong on routine digital activities does not imply strong on scientific tools — TUA-Bench makes this visible per family.

## Gotchas & tricks

- **Docker overhead is real.** Deterministic setup means containerised evaluation; expect per-task setup to dominate wall-clock for cheap agents.
- **PhD-designed tasks assume domain tooling.** Some scientific tasks require specific CLIs installed at exact versions — reproducing outside the paper's harness needs care.
- **65.8% is not the ceiling.** Human upper-bound isn't reported in the abstract; treat top-agent score as a *floor* on remaining headroom.

## Sources

- Paper: *TUA-Bench: A Benchmark for General-Purpose Terminal-Use Agents* — Wang et al. (Meta AI / Duke / Stanford), 2026 — [arXiv:2606.28480](https://arxiv.org/abs/2606.28480).
