# Code-World Simulation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Grow agents inside hybrid environments that pair a deterministic simulator or game engine (which owns rules and state) with a shared neural renderer (which paints each frame from the structured geometry). Agents observe only the rendered pixels and distill lessons across rounds into a reusable natural-language *playbook*, achieving in a handful of rounds what tabula-rasa RL needs millions of episodes for. Introduced in AgentGarten (2026).

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [rulebook-world-model.md](rulebook-world-model.md) · [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

A design pattern for agent training environments. Classical pure-sim RL puts the full world — physics, rendering, reward — behind one simulator. Code-world simulation splits those concerns:

1. **Rule engine** — code that holds persistent state and enforces interaction rules deterministically.
2. **Neural renderer** — a video-model-derived decoder that produces an RGB frame from the engine's structured geometry/scene conditions.
3. **Agent** — sees only the rendered pixels; it reasons, acts, writes down what it learned.

Across rounds, surviving agents compile their lessons into an inheritable textual *playbook*, so the next cohort starts from the collective state of the previous one.

## How it works

- **Renderer construction.** Start from a pretrained video model and distill it with **Adversarial Forcing**: a differentiable history-encoding scheme that replays past geometry exactly so the renderer learns to be consistent across time, combined with real-data adversarial supervision to keep photorealism. Then an inference-side optimization pass makes it fast enough for real-time interaction.
- **Observation loop.** At each step, the engine advances its structured state; the renderer paints the new frame; the agent emits an action; the engine applies it; repeat until episode end.
- **Playbook distillation.** At the end of a round, each agent compresses the trajectory into natural-language rules and heuristics ("the switch opens the door only after you pick up the yellow key"). The next cohort receives a merged playbook as part of its system prompt.
- **Population dynamics.** Multiple agents in parallel keep the search diverse; the playbook that survives is the one that generalizes.

## Why it matters

- Decouples *rules* from *appearance*: ground-truth state is cheap (the engine), and rich observations are reused across tasks (the renderer).
- Replaces per-task PPO/GRPO with *library-level* learning — the knowledge object is a human-readable playbook, not opaque weights.
- Reported sample efficiency: ~4 rounds of inter-generational refinement vs. millions of episodes for naïve RL baselines, under the environments studied.

## Gotchas & tricks

- The renderer–engine seam is where consistency bugs hide: if the renderer hallucinates objects that the engine does not track, agents learn ghosts.
- Playbook merging across generations is nontrivial — the paper leaves the choice of merge/consensus procedure open, and treating it as a reward-free SFT target is the dominant practical choice.
- Fast rendering matters more than photorealism; the inference optimization pass is load-bearing for closed-loop interaction.

## Sources

- Paper: *AgentGarten: Code Worlds for Evolving Agents* — Chi, Miao, Shi et al., 2026 — [arXiv:2610.12374](https://arxiv.org/abs/2610.12374).
