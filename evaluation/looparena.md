# LoopArena
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark that scores a model in the *Controller* role of a coding-agent loop: after each round it reads a structured run summary and tells a fixed *Worker* what to do or verify next, or when to stop. Splits evaluation into three tiers of execution cost — Type I (execution-validated MCQ over next-step "Loop Contracts"), Type II (repeated control over a slice of a full task), Type III (paired full runs) — so cheap Type I/II can proxy expensive Type III.

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** [livecodebench.md](livecodebench.md) · [humaneval.md](humaneval.md) · [README.md](README.md)

---

## What it is

The problem it names: in real coding-agent runs, the final outcome doesn't tell you whether the *loop* (the meta-agent monitoring progress, dispatching work, deciding to stop) is good or whether the *coder* is good. LoopArena decouples the two by freezing the Worker and only scoring the Controller.

## How it works

The Controller sees a **structured run summary** after each coding round (files touched, tests run, error trace, budget left, verifier signals) and picks a **Loop Contract** — a labeled next-step instruction from a fixed vocabulary (e.g. "run these tests", "revert last change and try X", "verify then submit", "stop, task complete"). Three evaluation tiers:

- **Type I — Contract selection MCQ.** Given a run state, pick the correct Loop Contract from execution-validated distractors. No Worker invocation at eval time — cheapest tier.
- **Type II — Sliced control replay.** Run the Controller over a fixed slice of a full task with the Worker in-loop. Cheaper than Type III, still measures dynamic control.
- **Type III — Paired full runs.** End-to-end from the original state. Reported as *Strict Success Rate*.

Also tracks estimated inference cost so a cheaper Controller that succeeds counts more.

## Why it matters

- Lets teams **shop for the Controller separately** from the Worker — pick a small/cheap model for loop control and route only the coding to a frontier model.
- Type II tracks Type III at ρ = 0.9747 under the paper's Core criterion, so most iteration can happen at Type II cost.
- Even top Controllers reach only **24.69%** Strict Success Rate on full tasks — long-horizon loop control is far from solved and there's substantial headroom.

## Gotchas & tricks

- Numbers are Worker-relative. Changing the fixed Worker changes the ceiling; report both.
- Loop Contract vocabulary is fixed — a Controller that reasons in a richer action space than the Contract set gets penalized on Type I.
- Paired inference-cost reduction averages **64.4%** across Controllers vs a strong baseline pairing; use this as a first-order screen when ranking.

## Sources

- Paper: *LoopArena: Benchmarking Models as Runtime Controllers for Loop Engineering* — Wang et al., Alibaba DreamX, 2026 — [arxiv](https://arxiv.org/abs/2608.28281)
- Code: https://github.com/AMAP-ML/LoopArena
