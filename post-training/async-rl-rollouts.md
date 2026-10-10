# Asynchronous RL Rollouts at ~1 M Context
*Depth — decoupling rollout production from policy updates so RL at million-token context stays throughput-bound, not blocking.*

**TL;DR:** Synchronous RL for LLMs blocks the optimiser on rollouts: no step until every rollout finishes. At 1 M-token context lengths and long-horizon tasks, that stall dominates. **Asynchronous RL rollouts** run rollout workers independently; the optimiser consumes trajectories from a queue as they land, with a bounded staleness. **MiMo-V2.6** (2026) reports asynchronous training that consumes **1,568 samples and 2.7–3.7 B tokens per RL step at context lengths up to 1 M**, enabled by decoupled control/data planes, training–inference consistency, and high-concurrency rollout.

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md), [_rl.md](_rl.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)
**Related:** [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

An execution pattern for RL post-training where rollout generation and policy optimisation are decoupled processes communicating through a bounded trajectory queue. The optimiser never waits for a specific rollout; it consumes whatever is on the queue, up to a maximum staleness bound.

Not just "generate in a worker" — asynchronous rollouts imply (a) an explicit queue abstraction, (b) a staleness budget enforced by the loss (importance sampling or clipped surrogate), and (c) infrastructure for trajectory streaming and reconciliation.

## How it works

Four infrastructure primitives:

1. **Unified trajectory representation.** All environments / harnesses write to the same schema (observations, actions, rewards, metadata). Letting the optimiser mix domains in a single step requires one trajectory type.
2. **High-concurrency rollout workers.** Many workers run in parallel, each with its own copy of the policy (potentially staler than the optimiser's), producing trajectories at full inference throughput.
3. **Decoupled control / data planes.** Scheduling (who rolls out what, which env to spawn) runs on one plane; trajectory bytes flow on another. Avoids head-of-line blocking.
4. **Training–inference consistency.** The rollout policy must match the training policy numerically — same quantisation, same kernels, same tokeniser path — else off-policy correction either over-corrects or lets drift accumulate silently.

The optimiser then applies a clipped off-policy loss (PPO/GRPO/ReSPO) that compensates for the staleness between the rollout policy and the optimiser's current policy.

## Why it matters

- **Context length no longer blocks.** A 1 M-token rollout can take minutes. Synchronous RL stalls the entire optimiser on the slowest trajectory; async hides that behind the queue.
- **Mixed-task RL stays dense.** Mixing code / general / visual / cyber domains in one step is only tractable when fast and slow rollouts don't stall each other.
- **Enables scaling along the "batch × context" axis.** MiMo-V2.6's 1,568-sample / 2.7–3.7 B-token / 1 M-context steps are credible only because async-queueing hides tail latency.

Generalises: the same pattern underlies agentic RL (long tool-use trajectories), partial-rollout systems ([partial-rollouts.md](../systems/partial-rollouts.md)), and distributed rollout farms.

## Gotchas & tricks

- **Staleness blows up without a budget.** Cap the oldest rollout the optimiser consumes (in gradient steps or wall time) and discard anything older — otherwise distribution shift kills learning.
- **Clipped surrogate matters.** PPO's clip handles mild staleness; heavy reuse needs reshaped-surrogate variants (α-divergence sequence-level objectives) to avoid gradient starvation.
- **Training–inference numerical drift** is the silent killer. If rollout and training paths differ in quantisation, the clip's importance ratio becomes meaningless and off-policy correction fails. Align kernels; verify with periodic on-policy ground truth.
- **Memory on rollout workers.** 1 M-context rollouts need KV-cache + weights on each worker. Hybrid sliding-window attention (as in MiMo-V2.6's base) bounds the memory cost; vanilla full attention does not.
- **Grader latency** can shift to being the bottleneck once rollouts are async. Group graders ([groupwise-agentic-grading.md](groupwise-agentic-grading.md)) help here.

## Sources

- Paper: *MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement* — Xiaomi LLM-Core Team, 2026 — [arXiv:2610.11959](https://arxiv.org/abs/2610.11959). States the per-step figures and names the four infrastructure primitives above.
