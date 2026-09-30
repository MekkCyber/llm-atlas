# Adaptive parallelism (Nereus)
*Depth — a cost-aware runtime that re-plans TP/PP/DP layouts mid-run for LLM RL post-training.*

**TL;DR:** RL post-training coordinates generation, inference, and training models on the same GPU cluster. Load, sequence length, and stage bottlenecks shift over the run, so a fixed parallelism plan becomes slow or infeasible. Nereus formalizes each model-in-a-stage replica as an **Elastic Model Unit** whose distributed state can be *transformed*, plans transitions against a cost model calibrated on the running job, and orders GPU transfers with a global transition graph.

**Prereqs:** [README](README.md), [../post-training/ppo.md](../post-training/ppo.md)
**Related:** [partial-rollouts](partial-rollouts.md), [ray](ray.md), [dualpipe](dualpipe.md), [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

A controller-plus-abstraction that turns "you pick one TP/PP/DP layout for the run" into "the runtime re-picks one whenever the job would benefit, without restarting." It targets RL post-training specifically because the workload's shape (generation vs inference vs training, sequence length, GPU availability) changes over a run in ways that a fixed layout cannot handle.

## How it works

- **Elastic Model Unit (EMU).** Each replica of a model-in-a-stage (e.g. "actor forward on stage 3") is represented as an EMU with its distributed state — shard layout, TP degree, PP schedule — as first-class metadata. State transforms (re-shard, re-partition) are defined as operations on the EMU, not full checkpoint reloads.
- **Cost-aware controller.** A low-overhead controller monitors bottlenecks and memory pressure. When a candidate new layout is memory-feasible, it estimates the transition cost against the currently running job (calibrated on real transfer bandwidth, not a static model) and admits the switch only if the projected win exceeds that cost.
- **Global transition graph.** Because many EMUs share GPUs, a naive transform-in-place would deadlock. Nereus builds a graph of the required transforms and inter-EMU transfers and executes them in a globally consistent order.

## Why it matters

Fixed parallelism plans have to be sized for the *worst* stage over the whole run, which means over-provisioning during easier stages. Nereus makes the parallelism plan a function of the current load, which is what production RLHF/GRPO stacks have needed to stop over-provisioning and to survive long-tail sequence-length changes over a training run. Reported wins: online TP/PP adaptation cuts average step latency by 27.7% vs a fixed initial layout; in a 1,000-step run scaling to 1,024 GPUs, six transitions consumed only 0.079% of total run time; end-to-end 8B PPO throughput is 2.14×–7.27× higher than OpenRLHF and 1.10×–1.47× higher than Verl.

## Gotchas & tricks

- **Cost-model calibration matters.** A wrong bandwidth estimate can trigger too-frequent transitions ("thrashing"). Nereus's controller filters candidates with a hysteresis threshold — implement the same in any port of the idea.
- **State transform ≠ checkpoint reload.** The point is to avoid a full serialize/deserialize; if your framework can't in-place resched TP/PP without touching the optimizer state, you don't get the wins.
- **Fault interaction.** Elastic parallelism composes cleanly with elastic *scale* (adding/removing nodes on preemption), but the two controllers must share the same global transition graph or they race.
- Best suited to RL post-training; ordinary pre-training doesn't shift shape enough for the transition overhead to pay off.

## Sources

- Paper: Nereus: Adaptive Parallelism for LLM Post-Training — Jiang, Shi, Zhang, Wang, Di Francesco, Zhao, 2026 — [arXiv:2609.34645](https://arxiv.org/abs/2609.34645)
