# TerminalHorizon
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An automated pipeline for building long-horizon agentic SFT trajectories for shell / terminal agents. Starts from real-world professional tasks, expands them stage-by-stage, rolls out a working agent, and keeps the trajectories that pass staged verifiers. Fine-tuning Qwen3.5-27B on the result gives gains across 12 agentic benchmarks and supports case studies of 12-hour autonomous research sessions.

**Prereqs:** [agents/README](../agents/README.md), [post-training/fine-tuning/README](../post-training/fine-tuning/README.md)
**Related:** [_data-curation](../data/_data-curation.md)

---

## What it is

A reproducible data engine for terminal-agent SFT. Produces trajectories that are *much longer* than those in existing agentic SFT corpora — multi-hour sessions with hundreds of commands, each verified against a chained goal.

## How it works

Stage-wise task expansion:

1. **Seed tasks.** Collected from real-world professional workflows (research scripts, system admin, data exploration).
2. **Decompose.** The pipeline splits each seed into subgoals; each subgoal becomes a shorter trajectory target.
3. **Agent rollout.** A working agent (any competent LLM in a shell loop) attempts the task; its command/output/stdout stream is captured.
4. **Staged verification.** Each subgoal's completion is checked against a bespoke verifier (file-state check, test suite, API inspection). Only trajectories that pass *every* subgoal verifier are kept.
5. **Trajectory stitching.** Verified subgoal trajectories are combined to produce the final long-horizon trace. Metadata about the subgoal structure is preserved.
6. **SFT target.** The combined trajectories are used to fine-tune the student (Qwen3.5-27B in the paper).

The staged-verifier pattern is key: at hour-long horizons, no single final-outcome check is reliable; chained local verifiers are.

## Why it matters

- **Agentic SFT data is the bottleneck** for terminal / shell / software agents. Hand-collection does not scale; synthetic pipelines without verification produce low-quality traces.
- **Long-horizon survival.** The paper demonstrates 12-hour autonomous research runs where the fine-tuned agent keeps making progress rather than getting stuck in loops — a direct capability improvement.
- **Reusable methodology.** The staged-expansion + staged-verifier recipe transfers to any domain with hierarchical task structure.

Reported: gains across 12 agentic benchmarks after SFT on TerminalHorizon data vs the Qwen3.5-27B baseline.

## Gotchas & tricks

- Verifier quality bounds trajectory quality. A noisy sub-goal verifier admits bad traces that pollute SFT.
- The pipeline is biased toward tasks where staged verifiers are *writable*. Pure exploration or open-ended research tasks are harder to include.
- 12-hour sessions in training don't guarantee 12-hour reliability in deployment — the paper's case studies are curated.
- SFT alone, without downstream RL, can plateau — the recipe is positioned as the SFT foundation, not the whole post-training stack.

## Sources

- Paper: *Scaling Agentic Data for Long-Horizon Terminal Intelligence* — Zou et al., USTC / AMAP Alibaba / SUSTech, 2026 — [arXiv:2610.13166](https://arxiv.org/abs/2610.13166)
