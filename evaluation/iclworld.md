# ICLWorld
*Depth — a benchmark that separates rule-recognition from rule-execution in ICL for agents.*

**TL;DR:** ICLWorld measures whether an agent can learn a task's rule from in-context demonstrations and *act* on it, in both digital (computer use) and physical (robot use) settings. 160 tasks across both domains; each task is given in three rule variants and in either text or video demos. Six frontier agents tested; the recurring failure is that agents correctly identify the rule in text but don't change their actions accordingly — rule recognition without rule execution.

**Prereqs:** [evaluation/README.md](README.md)
**Related:** [livecodebench.md](livecodebench.md) · [ifeval.md](ifeval.md) · [agents/README.md](../agents/README.md)

---

## What it is

A benchmark targeting the operational form of in-context learning that matters for agents: read a few demonstrations in context, infer the task's hidden rule, and then execute that rule in a world. ICLWorld separates agent domains (computer use vs. robot use), rule variants (three versions per task to avoid contamination), and demo modality (text vs. video). The point is to make it impossible to "look up" the answer from pretraining — the rule varies per instance, so the agent must actually use the in-context demo.

## How it works

Each ICLWorld task has four axes:

1. **Domain.** Computer-use (desktop apps, browsers) or robot-use (simulated embodied).
2. **Rule variant.** Three variants per task that differ only in the rule (e.g., "double-click opens / single-click moves" vs. "single-click opens / double-click moves"). Same physical world, different rule.
3. **Demo modality.** Text instructions or a short video demonstrating the rule.
4. **Execution.** The agent operates in the environment; success is measured by task completion.

Agents are scored on two decoupled metrics:

- **Rule recognition** — does the agent (in a text probe) correctly describe the rule after seeing the demo?
- **Rule execution** — does the agent *act* consistently with the rule in the environment?

160 tasks total; six frontier models were evaluated at release.

## Why it matters

- **Contamination-resistant ICL eval.** Rule variants make pretraining memorization useless. This is one of the cleanest ICL evals for agents at the moment.
- **Names a failure mode the field will target.** Rule recognition with no rule execution is a specific, learnable gap. Expect reasoning-RL and agent-training papers to use ICLWorld as the diagnostic.
- **Bridges digital and physical.** The 50/50 split across computer-use and robot-use exposes agents' asymmetric capabilities: all tested models do substantially better on computer tasks than on robot tasks.
- **Three rule variants per task.** A hedge against leakage — even if one variant appears in training data, the others are fresh.

## Gotchas & tricks

- **Rule recognition is text, rule execution is actions.** The two metrics come from different probes; don't collapse them. The whole point is the gap.
- **Robot-use results depend on the simulator.** The physical-world tasks run in a specific simulator; sim-to-real transfer isn't tested.
- **Not an RL training signal, per se.** The benchmark is for evaluation. You can train against it, but the three-variant design makes curriculum-RL non-trivial.
- **Base-model contamination is possible.** Three variants reduces, but doesn't eliminate, the chance that the exact rule-world combination was seen. Report per-variant breakdowns.
- **Video demos stress multimodal grounding.** Text demos and video demos are not equivalent — some models recognize rules from text but miss them from video. Treat as two sub-benchmarks for ablation.

## Sources

- Paper: *ICLWorld: Benchmarking In-Context Learning of Agents from Digital to Physical Worlds* — Li, Li, Zheng, Yang, Fan, Hong, Gao, Guo, Yang (Tsinghua), 2026 — [arXiv:2611.32649](https://arxiv.org/abs/2611.32649).
- Project site: iclworld.github.io.
