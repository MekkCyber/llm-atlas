# Horizon Scaling
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For agentic tasks, scaling an agent's *horizon* (tool-use depth, multi-turn context length, environment interaction breadth) matches or beats scaling raw parameters. Introduced as the design thesis of Agents-A1, a 35B-active MoE agent that reaches trillion-parameter-agent performance by training on **wider, deeper, more environment-grounded** trajectories rather than more weights.

**Prereqs:** [README.md](./README.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md), [../architectures/_moe.md](../architectures/_moe.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

Parameter scaling gives smooth returns on next-token prediction, but agentic tasks are *sequential decision problems* — the marginal benefit of a bigger model plateaus quickly once basic instruction-following works. Horizon scaling is the opposing thesis: allocate training budget to (a) longer trajectories, (b) more diverse environments, (c) more tool-call variety, and (d) more turns of teacher feedback along each trajectory. The Agents-A1 report claims a 35B MoE trained this way reaches or exceeds trillion-parameter frontier agents on long-horizon search, engineering, and science tracks.

## How it works

Three ingredients composed into one curriculum.

**Knowledge-action infrastructure.** A unified training substrate that couples tool-calling, environment stepping, and long-horizon search into a single trajectory format. The agent's action space includes tool calls, environment queries, and search operations as first-class tokens; SFT and RL both operate over full sequential traces.

**Full-domain SFT.** Rather than SFT on short instruction/response pairs, Agents-A1 uses full agentic trajectories across engineering tasks, scientific research, instruction following, and tool-calling. Trajectories are curated for depth (many turns) and diversity (many environments).

**Multi-teacher on-policy distillation.** The student rolls out; **multiple** frontier teachers score its trajectories — different teachers catch different failure modes (planning, stopping, tool choice, evidence use). The student is updated against the ensemble of teacher signals rather than a single teacher's KL. See [../post-training/on-policy-distillation.md](../post-training/on-policy-distillation.md) for the OPD substrate.

**Compute allocation shift.** Instead of doubling parameters, the same budget buys 2×–4× more agent-environment turns per training example. Empirically the return curve on trajectories is steeper than on parameters at this scale.

## Why it matters

- **Decouples agentic capability from parameter budgets.** A serious agent no longer requires trillion-parameter serving; 35B active is enough if the training curriculum matches the deployment surface.
- **Analogous to RLVR for reasoning.** RLVR showed reasoning capability scales with *rollout depth*, not parameters. Horizon scaling makes the same claim for tool-using agents.
- **Reprioritises infrastructure.** Rollout workers, environment fidelity, and trajectory curation become the frontier — the same trend RL-heavy labs have been signalling for a year.

## Gotchas & tricks

- **Environment diversity is the bottleneck.** Long trajectories in one environment give diminishing returns; the shape of the curve depends heavily on how many *distinct* action distributions the agent sees.
- **Multi-teacher disagreement needs adjudication.** The Agents-A1 report doesn't fully describe how conflicting teacher signals are combined (weighted vote vs sequential); expect this to be the next tunable knob.
- **Not a small-model story.** The 35B figure is *active parameters*; the total MoE count is much larger. Horizon scaling still assumes a competent base.
- **Evaluation coverage matters.** Reported parity with 1T agents is on tracks Agents-A1 was trained toward. Off-distribution agent tasks may reveal parameter-scale gaps.

## Sources

- Paper: *Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent* — Agents-A1 team, 2026 — [arXiv:2606.30616](https://arxiv.org/abs/2606.30616).
