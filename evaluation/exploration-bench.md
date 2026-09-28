# ExplorationBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark that measures whether an AI system can **discover new rules through exploration**, not merely recall pre-training knowledge. Built as two "Alien Worlds" whose rules are *executable* (so answers can be verified exactly) and *deliberately conflict with familiar priors* (so pattern-matching from pretraining fails). The system is given a flawed manual, task-specific environment feedback, and a tool-call schema; it must then explore, then solve held-out tasks.

**Prereqs:** [README.md](README.md)
**Related:** [../agents/README.md](../agents/README.md), [livecodebench.md](livecodebench.md), [mmlu.md](mmlu.md)

---

## What it is

Existing agent benchmarks reward pattern matching over exploration: BrowseComp-style deep-search tasks, coding benchmarks, and tool-use evals all skew toward "do you know how to do this?" ExplorationBench forces the opposite — **you have never seen these rules before**, and the only way to answer is to interact with the environment.

Two sandboxes:
- **AlienCode** — 31 discovery targets, 70 tasks.
- **AlienLogic** — 24 discovery targets, 70 tasks.

## How it works

Each world exposes:
- **Executable rules** — the sandbox implements a non-standard rule system (deliberately different from what a pretrained model would predict).
- **A flawed manual** — partial documentation that is enough to bootstrap exploration but not enough to solve tasks by reading.
- **Task-specific environment feedback** — after each interaction, the sandbox returns state or judgment.
- **A tool-call schema** — the interface the system uses to interact.

Systems have an exploration phase (bounded time / calls), during which they probe the sandbox. Then they solve held-out tasks; every answer is scored by executing the sandbox rules, so scoring is exact.

The two-property design (executable-and-verifiable + conflicting-with-priors) blocks both **contamination** (a familiar-looking task solved by recall) and **evaluation noise** (a judge model that can't decide).

## Why it matters

- Fills a real gap between "capability" evals (does the model know things) and "agentic" evals (can the model use tools); this one specifically measures whether the model can *acquire and apply new knowledge* in an unknown environment.
- The finding that continued exploration can *stall or reverse* prior gains is a substantive empirical result — current agents don't have stable exploration policies, only occasional lucky bursts.
- Directly targets the "scientific-discovery agent" narrative that has been under-specified.

## Gotchas & tricks

- Reproducibility depends on the sandbox implementation staying frozen; version pinning is essential.
- A model can still "solve" by lucky guessing on short tasks; the paper controls for this by reporting variance across trajectories.
- Priors-conflict is a limited resource: as more research on ExplorationBench-style tasks accumulates, models will start seeing analogues at pre-training time. Watch for contamination over subsequent releases.

## Sources

- Paper: *ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds* — Zhang, Xiang, Gao, Shen, Wang, Yue, Dou, Yin, Ye, Liu, Zheng, Chen, Chen, Liu, Shao, Gui, Zhang, Huang, Zheng, Pan — 2026 — [arXiv:2609.30199](https://arxiv.org/abs/2609.30199).
