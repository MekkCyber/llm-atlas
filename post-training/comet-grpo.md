# COMET-GRPO
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** GRPO assigns a single response-level advantage to every decision in the response. For hierarchical decisions (e.g. "pick a mode, then pick a model"), this smears the credit — bad decisions in one axis get rewarded alongside good decisions in another. COMET-GRPO decomposes the GRPO advantage into per-decision terms, so the policy learns which decision was responsible for the reward (NTU · HUST · UIUC, 2026).

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [_rewards.md](_rewards.md), [ppo.md](ppo.md)

---

## What it is

A credit-assignment refinement of GRPO for policies that make multiple structured decisions per rollout. Introduced to train a unified LLM router that must pick (a) an execution mode — single-round, multi-round, or agentic — and (b) a specific model within that mode. Treating these as one lump makes vanilla GRPO slow to learn either individual axis.

## How it works

- **Response decomposition.** Each rollout is split into named decision slots (mode-choice, model-choice, …). Each slot gets its own advantage term computed over the group of rollouts.
- **Per-slot normalisation.** Instead of one $(r - \bar r)/\sigma$ advantage per response, compute per-slot statistics across the group of G rollouts sharing the same prompt but differing in that slot's choice. Each decision then receives its own credit signal.
- **Progressive exploration.** A curriculum: early training assigns the mode (teacher-forced) and only the model-choice is trained; later, mode-choice is handed back to the policy. This stabilises the two-axis learning problem.
- **Rest of the loop is GRPO.** PPO-style clipped ratio, KL to reference model, no critic.

## Why it matters

- **Fixes a real learning pathology.** When mode-choice and model-choice are entangled, each axis slow-learns because the group noise from the other axis drowns its signal. Decomposing credit is the simplest fix.
- **Portable template.** Any hierarchical decision (routing, tool selection + argument, plan + act) can plug into COMET's decomposition.
- **No critic needed.** Inherits GRPO's value-network-free architecture; adds only bookkeeping, not compute.

## Gotchas & tricks

- **Slot definition is a design choice.** Decomposing too finely (per-token slots) regenerates PPO's per-token credit problem without the critic that handles it. Decomposing too coarsely regenerates vanilla GRPO.
- **Group composition matters.** Each slot needs variation across the G rollouts for the normalisation to be meaningful. If all G rollouts picked the same mode, mode-choice credit is undefined for that prompt.
- **Progressive-exploration schedule is sensitive.** Handing over mode-choice too early stalls; too late wastes the curriculum. The paper reports a tuned schedule.
- **Reward-function design still dominates.** Credit decomposition can't save you from a reward that doesn't distinguish good decisions from bad.

## Sources

- Paper: *RouteWeaver: Weaving Mode Selection and Execution into Unified LLM Routing* — Wang et al., NTU · HUST · UIUC, 2026 — [arXiv:2610.19705](https://arxiv.org/abs/2610.19705).
