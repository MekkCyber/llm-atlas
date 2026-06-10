# OmniGameArena
*Depth — a UE5 benchmark for VLM game agents with an improvement-dynamics metric.*

**TL;DR:** A real-time benchmark of **12 newly built Unreal Engine 5 games** for VLM agents, with **unified action interfaces** spanning **Solo (7), PvP (3), and Coop (2)** modes. Introduces the **Improvement Dynamics Curve (IDC)** — an agentic-reflection harness in which a tool-using reflector LLM autonomously refines a *bounded* skill prompt across multiple rounds, turning the headline metric from a single first-attempt score into a **trajectory of scores over reflection rounds**. Introduced by Lin et al. (HKU / LIGHTSPEED / CUHK / Tsinghua), 2026 (arXiv 2606.09826).

**Prereqs:** *(none)*
**Related:** [README.md](README.md) · [spatialworld.md](spatialworld.md)

---

## What it is

Game benchmarks for VLM agents historically report a **single first-attempt score per (agent, game) pair**, focus on Solo play, and lack a unified protocol across heterogeneous agents (commercial VLMs, open-weight VLMs, specialized policies).

OmniGameArena fixes those three:
- **Unified UE5 substrate.** 12 games built in Unreal Engine 5 with a single shared action API, so the *same agent* runs across all 12 with no per-game adapter.
- **Modes.** 7 Solo (single agent vs environment) + 3 PvP (agent vs agent) + 2 Coop (agents cooperate).
- **Agent classes covered fairly.** Commercial closed VLMs, open-weight VLMs, and specialized game policies all use the same interface — no asymmetry favoring incumbents.

## How it works as an LLM eval

- **Action interface.** Each game exposes the same set of high-level actions (move, look, interact, communicate). The VLM emits structured text; the game executes. Real-time — the agent has to act under a frame budget.
- **Improvement Dynamics Curve (IDC).** Around each (agent, game) pair, a **reflector LLM** is given the agent's recent trajectories plus a *bounded* skill prompt (capped in length so the reflector can't dump arbitrary context). The reflector edits the skill prompt — adding lessons, removing failed tactics. The agent plays again with the updated prompt. Repeat. The curve of score-vs-round is the headline metric.
- **Why bounded.** Without a budget, the reflector can grow the prompt arbitrarily, and "improvement" becomes a function of prompt size, not learning. The bound forces meaningful compression of lessons.

## Why it matters

- **Trajectory metric, not a snapshot.** First-attempt score conflates raw capability with how the agent learns in-context. The IDC separates them.
- **Agent classes on a fair footing.** Specialized game policies traditionally crushed VLMs on game benchmarks because they had bespoke interfaces. Unifying the interface re-levels the comparison.
- **Reflector LLM is the harness.** OmniGameArena treats *the reflection step* as the unit of agent intelligence — closer to how real deployment loops work than one-shot.
- **UE5 substrate.** Visually realistic, real-time, programmable. Lower sim-to-real gap than 2D / minecraft-style game benches.

## Gotchas & tricks

- **IDC saturates.** After a few rounds the bounded prompt fills up; further rounds plateau. The shape of the curve (slope, asymptote, rounds-to-saturation) is the comparison, not the final value.
- **Reflector quality bounds the IDC.** A weak reflector can't extract useful lessons even from rich trajectories. Use the same reflector when comparing agents.
- **Game balance matters for PvP / Coop.** Pairing affects scores. The paper randomizes; replicate that.
- **Real-time ≠ same frame budget for all agents.** A slow VLM can't keep up; a fast one can. Report wall-clock budget alongside score.
- **UE5 sim-to-real has limits.** Visually rich, but physics and environment dynamics are still simulation. Don't over-extrapolate to embodied real-world performance.

## Sources

- Paper: *OmniGameArena: A Unified UE5 Benchmark for VLM Game Agents with Improvement Dynamics* — Lin, Qian, Liu, Huang, Wang et al. — HKU / LIGHTSPEED / CUHK / Tsinghua, 2026 — arXiv 2606.09826.
