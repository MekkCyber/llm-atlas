# Self-Improving Agents

*Taxonomy — agents that convert accumulated execution experience into updates on themselves, spanning updates to model parameters and to the surrounding scaffold.*

**TL;DR:** A modern agent is a *(foundation model, operational scaffold)* pair, where the scaffold is prompts + memory + tools + control logic. "Self-improvement" is a self-induced *update operator* that consumes experience and writes updates back to either the model or the scaffold. Different published methods differ along two axes: **what they update** (parameters vs. scaffold components) and **what signal drives the update** (reward, self-critique, execution feedback, demonstration). This taxonomy uses that decomposition to organize the design space.

**Related taxonomies:** [_rl](../post-training/_rl.md) · [_post-training](../post-training/_post-training.md)
**Depth files covered here:** [gui-agents](gui-agents.md) · [on-device-agents](on-device-agents.md) · [failure-attribution](failure-attribution.md) · [agent-harness](agent-harness.md)

---

## The problem

A deployed agent accumulates trajectories continuously — successes, failures, edge cases, drift. The natural question is: *can it improve from this experience without a fresh human-labeled dataset?* Every candidate answer is a variant of the same operator — read the experience, decide what update to make, apply it — but the space of "what to update" and "how to derive the signal" is wide and under-organized. Without a taxonomy, comparing "self-improving prompt-search" to "self-improving memory-accumulation" to "self-improving RL fine-tuning" reduces to opinion.

## The shared pattern

Every self-improvement method has three parts:

1. **Experience source.** Trajectories, tool-call logs, user feedback, self-critiques, or execution outcomes.
2. **Update target.** One of: model parameters (weights), prompt template, memory store, skill library, control logic, tool set.
3. **Update signal.** How the experience is turned into a supervision signal — reward, gradient, retrieval-based rewrite, structured extraction, comparison against a demo.

A self-improvement method is characterized by the (target, signal) pair and its choice of experience source.

## Variants

| Variant | Update target | Signal | When it wins |
| --- | --- | --- | --- |
| RL post-training | Model parameters | Reward from execution outcome | Skills that require weight-level fluency; expensive setup, cheap inference. |
| Memory accumulation | Persistent memory | Successful trajectory extracts | Personal-assistant / long-horizon workflow reuse. |
| Skill library growth | Skill / tool set | Reflection on successful trajectories | Cross-platform GUI agents; recurring task patterns. |
| Prompt search / rewrite | Prompt template | Score of candidate prompts on eval set | Fast iteration; no weight update needed; brittle to shift. |
| Retrieval-augmented context | Memory store (retrieval) | Similarity + outcome | Domain adaptation without touching weights. |
| Control-logic rewriting | Control loop code | Self-critique / trace analysis | Under-explored — control logic is usually hand-written. |
| Tool augmentation | Tool set | Failure-mode analysis | When new capabilities are discoverable from failure traces. |

Link techniques with a depth file; leave others as plain text until a depth file lands.

## How to choose

**Start with the cheapest target** that plausibly fixes the failure mode:

- **New behavior needed once and reused often** → **skill library** or **memory**.
- **Systematic failure across many prompts** → try **prompt rewriting** first; only escalate to **RL fine-tuning** if the failure is truly at the weight level.
- **The tool interface itself is wrong** → **tool augmentation** / **control-logic edit**; no amount of RL will save a missing capability.
- **The signal is noisy or expensive** → keep updates off the weights; use retrieval + memory instead.

Combine freely: a well-designed agent updates its memory continuously, adds skills opportunistically, rewrites prompts on cadence, and RL-fine-tunes on a slower schedule.

## Adjacent but distinct

- **[_rl](../post-training/_rl.md)** — RL fine-tuning is *one* self-improvement variant (parameter target + reward signal), not the whole class.
- **[failure-attribution](failure-attribution.md)** — Attribution identifies *where* the update should apply; self-improvement decides *what* update to make. They compose.
- **Recursive self-improvement (RSI, unbounded).** The philosophical / long-horizon version, distinct from the incremental practical operator described here. The literature this taxonomy covers is deliberately bounded.

## Sources

- Survey: *Self-Improvements in Modern Agentic Systems: A Survey* — Ren, Chen, Guo, Rong, Li, Xiong, Lan, Wang, Nanbo, Yang, Zhuge, Schmidhuber, 2026 — [arXiv 2607.13104](https://arxiv.org/abs/2607.13104). Provides the *(foundation model, scaffold)* framing and the update-target × signal taxonomy adopted here.
- Related: *Know Deeply, Act Perfectly: Personal GUI Assistant with Self-Evolving Memory and Skill* — Li et al., 2026 — [arXiv 2607.12625](https://arxiv.org/abs/2607.12625). Memory + skill-library variants realized in a GUI agent.
