# Coding-Agent Harness Design
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A coding agent's "harness" is the code around the model that turns its outputs into a working software-engineering trajectory: the execution loop, the action space, the planning layer, and the context-management strategy. A 2026 empirical study fixes the loop and ablates the other three across 176 matched settings on SWE-Bench Verified and Terminal-Bench 2.1, distilling four findings: context management prevents overflow failures (mostly), rule-based elision beats LLM summarization on efficiency, planning shifts from an accuracy scaffold to a cost saver as models get stronger, and predefined tools help weak-bash models but bash-capable models are cheaper on a bash-only interface.

**Prereqs:** [README.md](./README.md)
**Related:** [../evaluation/livecodebench.md](../evaluation/livecodebench.md) · [../evaluation/humaneval.md](../evaluation/humaneval.md)

---

## What it is

A coding agent decomposes into four modular concerns, each with its own design axis:

1. **Execution loop.** How outputs become tool calls, how results are fed back, and how termination is decided. Usually kept fixed while other axes vary.
2. **Action space.** What primitives the agent can call. Two poles: **predefined tools** (file edit, shell, search) with typed schemas, or **bash-only** (a single tool that runs a shell command).
3. **Planning.** Whether the agent produces an explicit plan before acting. Ranges from none, through one-shot plan, to interleaved plan-and-execute.
4. **Context management.** How long-running trajectories are kept within the context window. Strategies: **rule-based elision** (regex-driven truncation of stale tool outputs), **LLM-based summarization** (a smaller pass compresses old turns), **hybrid** (elide first, summarize survivors), and **recoverable elision** (the agent can un-truncate an elided block).

## How it works

The referenced empirical study fixes the loop, varies the other three axes across four models on SWE-Bench Verified (176 matched settings) and Terminal-Bench 2.1, and measures accuracy and token cost per trajectory. Trajectory-level analysis explains the mechanics:

- **Context management** primarily *extends trajectories* — it doesn't change what the agent decides to do at any given step; it delays context overflow.
- **Planning** primarily *changes where trajectories stop* — planned trajectories terminate sooner (or later) depending on whether the plan reveals dead ends.
- **Action space** primarily *changes the granularity at which code is written* — bash-only agents write short shell one-liners; predefined-tool agents write per-file edits.

## Why it matters

- **Turns harness engineering into a matrix.** The paper's four findings replace folklore with measured tradeoffs. If you're choosing components, this is the reference.
- **Kills the "recoverable elision" pattern.** Making elided content recoverable is machinery models rarely use and yields no accuracy gain. Drop it.
- **Right-sizes the planner.** Strong models don't need planning as an accuracy scaffold, but planning still saves cost by pruning trajectories. Weak models need both.
- **Action-space depends on the model.** Weak-bash models need typed tools; bash-capable models are cheaper on bash-only, especially on command-line-centric tasks.

## Gotchas & tricks

- **Context management wins scale with budget tightness.** With generous context windows the benefit shrinks; most of the gain comes from preventing overflow failures. Budget-aware harness design isn't monotonic.
- **"Better" summarization can hurt.** LLM-based summarization introduces a compression LLM whose own noise enters the trajectory. Rule-based elision + LLM summarization *of survivors* dominates on efficiency.
- **Planning failure mode.** A planner that hallucinates a bad plan will lock the agent into a longer wrong trajectory. Test with random-seed variance, not just single trajectories.
- **Bash-only is not equivalent to no tools.** The action space is still huge (every command), but the agent must know shell. Rank models on `bash proficiency` before deciding.
- **Extrapolation risk.** Findings are from SWE-Bench Verified + Terminal-Bench 2.1 with four models; other domains (web agents, database tasks) may have different sweet spots.

## Sources

- Paper: *An Empirical Study of Harness Design for Coding Agents* — Fan et al., UMass Amherst + Tencent AI Lab, 2026 — arXiv:2609.20804.
- Related: *SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness* — Liu et al., NVIDIA + HKUST + MIT, 2026 — arXiv:2609.20519. Recursive-self-improvement approach that evolves harness components (action execution, context compaction, observation handling, delegated reading) automatically.
