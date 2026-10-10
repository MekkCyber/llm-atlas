# Groupwise Agentic Grading
*Depth — a grader design for long-horizon agentic RL that scores grouped rollouts jointly for more accurate rewards and shorter solutions.*

**TL;DR:** Long-horizon agentic tasks are hard to grade per-step and expensive to grade with learned RMs. **Groupwise agentic grading** evaluates each rollout *in the context of its group* (the batch of rollouts for the same prompt), using an agentic grader LLM that can inspect, compare, and compose evidence across the group. Reported in MiMo-V2.6 (2026) as producing *more accurate rewards for long-horizon tasks* and nudging rollouts toward *shorter, more token-efficient* solutions.

**Prereqs:** [_rewards.md](_rewards.md), [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [cot-reward-model.md](cot-reward-model.md) · [reasoning/orm.md](reasoning/orm.md) · [reasoning/prm.md](reasoning/prm.md)

---

## What it is

A grader design that sits above the per-rollout verifier. Standard RLVR grades each rollout independently against a rule (did the final answer match, did the code pass tests). Groupwise agentic grading **grades the group**: for each prompt and its $G$ rollouts, an agentic grader LLM looks at all $G$ trajectories, extracts their decisions, and assigns rewards that depend on both the trajectory *and* its relationship to the others.

Natural fits for the group-view include:

- **Relative length / efficiency.** "This correct rollout used half the tokens of the others — reward more."
- **Partial credit on long-horizon.** "These three trajectories all made the right first move but diverged; grade on the shared early reasoning."
- **Hack cross-checking.** "This rollout answered correctly but did it by exploiting an environment bug; penalise."

## How it works

For each prompt $x$ with rollouts $\{o_1, \ldots, o_G\}$:

- An agentic grader (LLM + tool access) reads the trajectory bundle.
- It executes a grading plan: checks correctness, measures length/efficiency, flags suspicious patterns, scores sub-steps against the group's median.
- It emits per-rollout scalar rewards $r_1, \ldots, r_G$ that are then fed into the usual GRPO baseline / clipped update.

Because the baseline in GRPO is already the group mean, grading at the group level composes naturally with the RL algorithm: scores can be shaped against the group's distribution without additional normalisation gymnastics.

## Why it matters

- **Fits long-horizon tasks** where no cheap rule verifier gives useful per-step signal. Agent trajectories, software-engineering rollouts, multi-tool search traces — all of these are easier to grade *comparatively* than *absolutely*.
- **Natural length-efficiency pressure.** Comparing trajectories inside a group makes "shorter correct is better" a direct observation, not a hand-tuned length penalty.
- **Hack visibility.** A reward-hacking strategy that looks fine in isolation often stands out when compared to peer rollouts. The grader can flag it from the group view.
- **Compatible with GRPO.** No change to the optimiser; the grader just produces richer per-rollout scalars.

## Gotchas & tricks

- The grader itself can be reward-hacked. Mix multiple grader models or run a periodic human spot-check.
- Group size $G$ interacts with grader cost — larger groups give the grader more comparison signal but multiply per-step grading tokens. MiMo-V2.6 doesn't disclose its exact $G$; typical GRPO ranges apply (8–64).
- Latency is a bottleneck under async rollouts — grader throughput must keep up with the rollout queue. See [async-rl-rollouts](async-rl-rollouts.md).
- Comparative scoring risks collapsing to "winner takes all" rewards; keep per-rollout absolute components (correctness, verifier) in the composed reward so the group signal is a modulator, not the sole signal.
- Avoid letting the grader itself re-use the policy under test as its judge — it leaks exploits.

## Sources

- Paper: *MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement* — Xiaomi LLM-Core Team, 2026 — [arXiv:2610.11959](https://arxiv.org/abs/2610.11959). Names "groupwise agentic grading" as the grader design and states the long-horizon-accuracy and shorter-solution effects.
