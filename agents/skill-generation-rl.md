# Skill-Generation RL (Skill-α)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reinforcement learning for **authoring** agent skills (not using them). Skills have no direct correctness label — you only find out if a skill is any good by running an agent with it. Skill-α treats the skill-writing LM as a policy, deploys each candidate skill in downstream tasks, and uses the resulting agent-success delta as the reward. A progressive curriculum lets later skills compose earlier ones.

**Prereqs:** [agent-skills](agent-skills.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/ppo.md](../post-training/ppo.md)

---

## What it is

Two orthogonal problems in the Skills lifecycle:
- **Skill use** — given a bank of skills, train an agent to apply them correctly. (See SKT.)
- **Skill authoring** — given raw documents or agent-experience logs, *generate* the skills themselves.

Skill-α tackles the second. Prior work has relied on heuristic pipelines ("consolidate this trajectory into markdown") or per-source hand-designed templates. The learning-based route was blocked because there's no ground-truth reward for a skill in isolation — only for whether it makes an agent better downstream.

## How it works

Skill-α frames skill authoring as an RL problem where the reward comes from downstream deployment:

```
1. Sample a candidate skill s ~ π_θ(· | source_material)
2. Deploy an agent equipped with skill s on a bank of tasks
3. Reward = (agent success rate WITH s) − (agent success rate WITHOUT s)
4. Update π_θ with an RL algorithm (GRPO-style rollouts over skills)
```

The **progressive** part: skill authoring is trained in curriculum rounds. Round-$k$ tasks are chosen to be solvable with round-$(<k)$ skills as building blocks, so the policy learns to compose earlier skills into later ones instead of re-inventing them.

**Reward attribution.** Because a skill can affect many downstream tasks, per-skill reward is aggregated over a small task batch — noisy but unbiased.

## Why it matters

- **First learning-based recipe** for skill authoring that beats heuristic consolidation baselines.
- **Complementary to SKT** — SKT trains use, Skill-α trains authoring. Full loop: better authored skills → more useful skill bank → better trained users.
- **Environment-based reward.** No human labels for individual skills; downstream success is the label. Fits the RLVR / verifiable-reward pattern.
- **Curriculum by construction.** Progressive rounds let the model learn compositional skill families without an explicit skill-graph design.

## Gotchas & tricks

- **Reward variance is high.** A single skill's contribution to a single task is noisy. Average over 10+ downstream trials per skill during rollout.
- **Reward hacking.** A candidate skill that hard-codes downstream task answers gets high reward but is useless in the wild. Include an "unseen-task holdout" evaluation in the reward to catch this.
- **Curriculum design matters.** Round-1 tasks must be solvable by *primitive* skills; round-2 tasks by primitive + round-1 skills; etc. Getting the difficulty gradient right is where most of the manual tuning lives.
- **The bank grows unboundedly.** Prune skills that are dominated by newer variants — otherwise the discovery cost during rollout blows up.

## Sources

- Paper: *Progressive Agent Skill Generation via Reinforcement Learning* — Shen, Zhang, Guo, Cheng (CUHK / Lightspeed), arXiv:2608.01678, 2026.
