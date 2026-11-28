# OS-Omni
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A live, final-state-scored benchmark for generalist computer-using agents spanning six platforms (Windows, Ubuntu, macOS, iOS, Android, Web) plus 12 cross-platform desktop-phone workflows. Tasks are long — humans need ~67.5 interaction steps vs. 15 in OSWorld — and frontier agents trail humans by ~9 points on average, with the gap concentrated on iOS and macOS.

**Prereqs:** [../evaluation/README.md](../evaluation/README.md)
**Related:** [../evaluation/livecodebench.md](../evaluation/livecodebench.md), [../evaluation/ifeval.md](../evaluation/ifeval.md)

---

## What it is

A benchmark for computer-using agents — agents that control a real OS by reading screenshots and emitting mouse/keyboard events. Unlike OSWorld (single-platform, short tasks) and WebArena (web-only), OS-Omni evaluates platform *generalization* and long-horizon state tracking on the same task families run across every OS, scored by whether the final system state matches the goal.

## How it works

- **Scope.** v1 has 300 tasks across six platforms and 12 cross-platform workflows (e.g. receive a code on an Android phone, paste it on a Mac).
- **Scoring protocol.** Final-state checks: after the agent reports completion, the harness reads system state (files, registry, cloud artifacts) and compares to a goal snapshot. No action-trace matching, so agents can take any path to the goal.
- **Reference humans.** Real users completed the tasks, needing ~67.5 interaction steps per task on average.
- **Live environment.** Not a canned trace replay — the agent acts in a VM / device in real time, which allows the benchmark to include tasks whose correctness depends on real services (email, cloud sync).

## Why it matters

Three things previous computer-use benchmarks couldn't do, this does:

1. **Cross-platform evaluation.** The same task family on Linux, macOS, Windows, iOS, Android, and Web lets you see where an agent's GUI priors break down. Claude Opus 5.5 beats humans on Linux but trails by >25 points on iOS/macOS — the model, not the task, is platform-biased.
2. **Long-horizon measurement.** 67.5 steps/task vs. OSWorld's 15 — long enough that state tracking, not pixel grounding, becomes the bottleneck.
3. **Cross-device coordination.** 12 workflows that require a phone and a desktop to cooperate — a class of task nothing else benchmarks.

## Gotchas & tricks

- Final-state scoring is strict: an agent that completes the goal and then takes one extra destructive action can fail. Some teams report two numbers (strict final-state, and "best observed state during run").
- Live services drift. The paper's reported numbers tie to specific cloud account states; replication requires setting those up.
- The cross-platform gap is partly a tokenizer / screenshot-grounding issue: agents trained mostly on Windows screenshots underperform on retina-DPI macOS captures. Not purely a reasoning gap.
- Cross-device workflows need both an iOS/Android simulator and a desktop VM — replicating the 12 cross-platform tasks takes more infrastructure than the single-OS ones.

## Sources

- Paper: OS-Omni: Benchmarking Generalist Computer-Using Agents on Long-Horizon Workflows Across Platforms — Shen et al. (Touchstone AI et al.), 2026 — [arXiv:2611.34232](https://arxiv.org/abs/2611.34232)
