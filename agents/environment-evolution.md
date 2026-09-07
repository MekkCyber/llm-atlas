# Environment Evolution
*Depth — growing agent-training environment difficulty off-policy across training generations.*

**TL;DR:** Once frontier agents saturate a fixed pool of executable environments, the learning signal collapses. Environment Evolution (2026) grows env difficulty **off-policy** using three evolution directions derived from the multi-turn RL objective, and schedules the evolved cohorts generation-by-generation, keeping the difficulty curve just ahead of the policy. On Qwen3.6-27B / Qwen3.6-35B-A3B, simple long-horizon RL on evolved envs adds +14.4 / +18.0 pts on Terminal-Bench 2.1 respectively.

**Prereqs:** [rlvr](../post-training/rlvr.md), [rl-prompt-curation](../post-training/rl-prompt-curation.md)
**Related:** [trajectory-to-environment](trajectory-to-environment.md), [_agent-training-environments](_agent-training-environments.md)

---

## What it is

A curriculum mechanism for agent RL: a *fixed generator* pipeline that ingests existing environments and emits harder variants, without using the trained policy's own rollouts to steer the generation. Envs from generation $g$ are scheduled into training generation $g$; new evolutions are produced ahead of time.

Contrast with **co-evolution** (Absolute Zero, EnvGen, and similar): those close the loop by generating envs specifically against the current policy's weaknesses, which limits generalization and stalls once the model's on-policy weaknesses shrink.

## How it works

From the multi-turn RL objective, three directions provably increase environment difficulty:

1. **Longer horizons.** Add required sub-tasks, chain intermediate goals, deepen the tool-call sequence needed to solve.
2. **More alternatives to reject.** Introduce plausible-looking wrong branches (distractor files, mislabeled configs, red-herring APIs) so the policy must discriminate.
3. **Higher inference / recall load.** Constrain the policy to remember state across steps (e.g. remove intermediate progress markers, require deferred consistency).

These directions are implemented through a loop-engineered multi-agent harness: mutation agents propose evolutions along each direction, an executor agent runs the mutated env end-to-end to check it still boots and its intended task is still solvable, and a difficulty-check agent (or a frontier LLM proxy) confirms it is genuinely harder.

Cohorts are then scheduled into generations — g=1 is baseline, g=2 evolves g=1's env pool along the three directions, g=3 evolves g=2, etc. — feeding training with progressively harder environments.

## Why it matters

- **Off-policy = generalizable.** Envs evolved off-policy do not overfit to any one model's rollout weaknesses, so the same evolved pool trains multiple policies well.
- **Sustained signal.** Where fixed pools saturate and co-evolved pools stall with the model, generation scheduling keeps the difficulty curve just above the policy's current capability.
- **Complements Terminal-Universe.** Trajectory-to-env supplies breadth; env evolution supplies depth in difficulty.

## Gotchas & tricks

- **Verify mutated envs boot.** A mutation that breaks the runtime yields zero learning signal, not a hard task.
- **Difficulty judge quality.** The frontier LLM used as difficulty proxy (Hy4 preview / Claude Opus 5 / GPT-5.6 Sol in the paper) must itself struggle with the evolved env — otherwise the "harder" label is unreliable.
- **Do not mutate the reward.** Only the environment surface (files, tools, transcripts) changes; the verification predicate for the underlying task stays fixed so evolved envs remain RLVR-verifiable.
- **Interleave generations.** Training only on the latest generation collapses breadth; mix in prior generations at diminishing rates.
- **Budget-vs-freshness.** Producing each new generation costs multi-agent compute; balance the frequency of new-generation drops against the training-step budget.

## Sources

- Paper: *Environment Evolution for Terminal Agents* — Fan et al., 2026 — https://arxiv.org/abs/2609.04128
- Related: *Terminal-Universe* — Wu et al., 2026 — supplies the initial env pool that this evolves.
