# WeaveBench

*Depth — a long-horizon, real-world benchmark for computer-use agents that must coordinate GUI observations with CLI / code operations inside a unified workflow.*

**TL;DR:** Real production work doesn't split cleanly into "GUI agent" or "code agent" — most useful workflows involve both. WeaveBench is a 114-task, 8-domain benchmark where each task requires an agent to **interleave GUI clicks with CLI commands or code execution**, evaluated with **trajectory-aware judging** to prevent reward hacking. Frontier-tier Claude Opus 4.7 hits **35.1%** PassRate on the fixed OpenClaw harness — substantially below its single-channel performance, exposing a long-horizon orchestration gap that GUI-only (OSWorld) and CLI-only (SWE-bench) benchmarks don't measure.

**Prereqs:** [README](README.md), [../agents/README](../agents/README.md)
**Related:** [livecodebench](livecodebench.md), [humaneval](humaneval.md)

---

## What it is

A benchmark for **hybrid-interface** computer-use agents. Each task is a multi-step real-world workflow that requires the agent to:

- Read or interact with a GUI (browser, application, file picker).
- Run CLI commands or write/execute code (shell, Python, SQL).
- Pass information between the two surfaces — e.g. read a value from the GUI, transform it in code, paste it back.

Tasks span 8 domains (e.g. data analysis with a notebook + database UI, infra ops with a dashboard + shell, document generation with an editor + scripting). Total: 114 tasks. The harness is **OpenClaw**, a fixed agent runtime that provides the agent with both screen / DOM observations and a shell / code-exec tool channel.

The judging is **trajectory-aware**: not just "did the final state match the expected output" but "did the trajectory match the expected workflow shape," to detect short-circuits (e.g. an agent that figures out the answer through pure code without ever touching the GUI on a GUI-required step, or vice versa).

## How it works

- **Task definition**: each task ships with a starting environment snapshot (VM image + browser state + filesystem) and a structured expected-outcome spec.
- **Runtime**: OpenClaw harness exposes synchronized GUI + CLI tools. The agent emits structured tool calls and observes both screenshots and command output.
- **Trajectory judging**: a judge inspects the trajectory log (tool calls, intermediate states) and scores against expected workflow constraints — *both* the final state and the path matter. This is the explicit anti-reward-hacking measure: an agent that hits the right final answer via the wrong workflow doesn't get full credit.
- **PassRate** is the headline metric: fraction of tasks where the agent's full trajectory + final state both pass the judge.

## Why it matters

- **Exposes a gap that solo-modality benchmarks miss.** A model can be SOTA on SWE-bench and OSWorld separately and still flounder when a real task requires bouncing between the two surfaces — Opus 4.7's 35.1% headline is the empirical proof.
- **Trajectory-aware judging is a generally useful eval design.** Reward hacking via final-state-only judging has been a known failure mode of agent benchmarks; WeaveBench's pattern (judge the path, not just the destination) is reusable.
- **Realistic** — task domains and interface mixes were drawn from real workflows, not adversarial constructions.

## Gotchas & tricks

- **Harness coupling matters.** WeaveBench's 35.1% number is on the fixed OpenClaw harness; the same model in a different harness can show different gaps. The benchmark measures the *model + harness* combination, not the model alone.
- **Trajectory judging needs a workflow spec.** Building the spec is the labor-intensive part of authoring a task; without it the judge can't detect short-circuits.
- **No partial credit by default.** A task that's 90% correct gets 0; the binary PassRate metric is harsh but matches production use.
- **Domain coverage**: 8 domains is broad but not exhaustive. Specialized hybrid workflows (e.g. CAD + code) aren't represented.

## Sources

- Paper: WeaveBench — Li et al. (2026) — [arXiv:2606.09426](https://arxiv.org/abs/2606.09426)
