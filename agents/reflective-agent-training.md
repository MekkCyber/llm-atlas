# Reflective Agent Training (Agent-Reflex)
*Depth — a three-stage recipe that trains language agents to detect bad tool feedback and recover without restarting.*

**TL;DR:** Real-world agent trajectories are full of noisy tool outputs — truncated results, 500s, hallucinated search hits. Agents trained on clean traces fail silently on these: they *trust* the bad result and chain further errors on top of it. **Agent-Reflex** treats noisy tool feedback as a first-class training signal and trains the agent to notice, reflect, and amend its plan mid-trajectory. The recipe is three stages: skill discovery → reflective SFT → on-policy reflective policy optimization.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [progress-reward.md](./progress-reward.md) · [recursive-harness-refinement.md](./recursive-harness-refinement.md) · [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

A supervised + RL recipe that embeds explicit **reflection turns** inside multi-turn tool-use trajectories. A reflection turn is a step where the agent, before acting again, verbalizes whether the last tool result is trustworthy and whether its current plan should be amended. Training optimizes both reflection quality (via SFT) and the downstream task reward conditioned on reflecting (via RL).

## How it works

**Stage 1 — reflection-skill discovery.** Collect rollouts from a base agent on noisy environments (injected tool failures, truncated outputs, decoy results). Keep trajectories where the agent eventually recovered. Diff recovered vs. non-recovered trajectories at the step preceding recovery and *extract the discriminating pattern* — e.g. "notice that the search result doesn't contain the queried term," "notice that the file written in step k wasn't in the listing in step k+1." These become the catalog of reflection skills.

**Stage 2 — reflective SFT.** Convert the recovered trajectories into supervised data by inserting an explicit `<reflect>...</reflect>` turn between the bad tool response and the corrective action. Fine-tune the base agent on this augmented data. The model learns to produce the reflection turn in the right places.

**Stage 3 — reflective policy optimization.** Run RL (GRPO-style) with the agent now sampling reflection turns on its own. Reward is terminal task success, but the training distribution is enriched with noisy environments so trajectories that correctly reflect-and-recover get reinforced over those that don't.

## Why it matters

- Shifts noise-robustness from prompt engineering to training. Prompt-level "double-check your tool output" instructions are brittle; a trained reflection habit survives distribution shift.
- Measurable: **Qwen3-8B → 58.6 overall across five long-horizon tool-use benchmarks** and **29.1 on AgentNoiseBench**, outperforming vanilla agentic SFT and plain RL baselines.
- Composes with other agent-training work (dense progress rewards, recursive harness refinement).

## Gotchas & tricks

- **Skill discovery is the bottleneck.** If the diffing step misses the discriminating pattern, SFT data has reflection turns in the wrong places; the model learns to reflect randomly.
- **Reflection-turn bloat.** Without a length penalty, the agent learns to always reflect, slowing inference. Give the policy a small per-token cost in the RL reward.
- **Noise-training distribution needs variety.** Training only on truncation noise doesn't generalize to decoy noise and vice versa; mix.
- **Base-agent competence floor matters.** Reflection on a very weak base yields "notice everything, fix nothing" trajectories — mostly noise.

## Sources

- Paper: *Agent-Reflex: Teaching Language Agents to Act Reflectively* — Yang, Jin, Hu, Song, Chen, Li, Tan, Zhu, Chua, Wen, Dou — Renmin U. / NUS, 2026 — arXiv:2610.17953.
