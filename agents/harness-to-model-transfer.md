# Harness-to-Model Decision Transfer
*Depth — distilling an orchestration harness's branching logic back into the base model.*

**TL;DR:** Computer-use agents (CUA) commonly run inside an external harness that branches on model outputs — tool selection, replan decisions, early-stop. Harness-to-model decision transfer logs the harness's branching during inference and uses the logs as supervision to fold the branching logic into the model itself. Future runs rely less on the harness; eventually the model can run with a thinner harness or none. OpenGCUA uses this to combine GUI and programmatic (code) action paths in one agent.

**Prereqs:** [README.md](README.md), [post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

A specific flavor of behavior cloning aimed at a specific target: the harness's control-flow decisions, not just its final outputs. The harness is treated as an expert policy over structured decisions; the model is distilled to match that policy.

## How it works

1. Run the agent with the full external harness on a task distribution. For each decision point (choose tool, retry, abort, switch between GUI and code path), log the harness's choice and the context the harness saw.
2. Format each (context, decision) pair as a supervised example — the context becomes the input, the decision becomes a structured token in the output (e.g. `<tool>gui</tool>` vs `<tool>python</tool>`).
3. SFT the base model on this corpus, mixed with its usual action-generation data.
4. Deploy with a reduced harness; branching now happens inside the model's generation.

## Why it matters

Harnesses are the main source of brittleness and lock-in in agent stacks: upgrading the model usually means re-tuning the harness. Folding decisions in makes model upgrades translate to behavior upgrades without harness changes, and shortens tool-use chains at inference (programmatic path short-cuts long click sequences).

## Gotchas & tricks

- Only transfers decisions the harness actually makes. Un-logged decisions (silent defaults, hardcoded limits) don't move.
- Loss weighting matters: naively mixing decision tokens with action tokens can under-train decisions since they're rare. Up-sample.
- As the model absorbs harness logic, failure modes shift inside the model — harder to debug, since you can't just read the harness source. Keep a lightweight audit harness for logging.

## Sources

- Paper: *OpenGCUA: Toward a Fully Open Stack for Generalist Computer Use* — Zhou et al., Alibaba MAI-UI, 2026 — [arXiv:2610.22200](https://arxiv.org/abs/2610.22200)
