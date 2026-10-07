# OSWorld-Pro
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A process-based successor to OSWorld for evaluating computer-use agents (CUAs). Where OSWorld grades only the end-state deliverable after hundreds of steps, OSWorld-Pro adds **subgoal-level grading**: 300+ tasks, 2,800+ subgoals, and 67k human annotations give an LLM judge the signal to say *where* a trajectory went wrong, not just whether it finished. Introduced by NVIDIA (2026).

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../agents/README.md](../agents/README.md) · [aime.md](aime.md) · [humaneval.md](humaneval.md)

---

## What it is

Computer-use agents operate for hundreds of steps per task. OSWorld (the predecessor) runs a functional verifier against the final filesystem/UI state. The verifier returns a single bit — "deliverable matches" or not — which gives no information about **which** step broke the trajectory.

OSWorld-Pro replaces that single bit with a sequence of **subgoal checks** along a canonical solution path, each graded by a human-aligned LLM judge. The result is a per-task fulfillment curve showing how far the agent got before failing.

## How it works

- **Task structure.** 300+ tasks spanning desktop apps (browsers, office suites, system settings). Each task is decomposed into a sequence of **sequentially dependent subgoals** (median ~9 subgoals/task).
- **Annotation.** ~67k human annotations label canonical subgoal outcomes: what the UI state should look like after each subgoal, and what the agent's intermediate actions must achieve.
- **Judging.** Each subgoal is graded by a robust LLM judge calibrated against human agreement. The judge is "process-based" — it reads the agent's action log plus environment state and decides whether the subgoal is fulfilled, not just whether the final deliverable is correct.
- **Scoring.** Report fraction of subgoals fulfilled per task, plus failure-category rollups (click-based error, keyboard-input error, subgoal-irrelevant action, …).

## Why it matters

End-state CUA benchmarks saturate while the models still fail *interestingly* mid-trajectory. Claude Opus 5 scores **75.7% on OSWorld-Pro vs. 83.4% on OSWorld**, so process grading is strictly harder and strictly more informative. Failure categories let a researcher triage: a model that errs on keyboard input needs a different fix than one that misclicks. This is the diagnostic signal needed to move CUAs forward past the "it mostly works" plateau.

## Gotchas & tricks

- **Judge calibration is load-bearing.** Process-based grading inherits any bias in the LLM judge. The paper reports human-agreement calibration; downstream users should re-check for new agent architectures that may fool the judge in unseen ways.
- **Subgoal sequences assume a canonical path.** Agents that solve a task by an alternative valid route may be scored low on fulfillment while actually succeeding at the end state. Report both metrics.
- **Not for benchmarking tool availability.** The suite assumes the standard OSWorld environment; it measures agent competence, not tool-chain robustness.

## Sources

- Paper: *OSWorld-Pro: Process-based Evaluation for Computer Use Agents* — Zhang et al., NVIDIA, 2026 — [arXiv:2609.24890](https://arxiv.org/abs/2609.24890).
- Predecessor: *OSWorld* — Xie et al., 2024 — end-state CUA evaluation.
