# Contextual Density Reward (LongR)
*Depth — a dense RL utility signal for long-context reasoning computed as model-measured information gain over consulted document spans, no evidence annotation required.*

**TL;DR:** Long-context reasoning RL has stalled on two bottlenecks: sparse terminal rewards and the cost of annotating which document spans are "evidence." **LongR** fixes both by letting the model itself measure utility. For each span the agent chooses to consult, LongR computes the **relative information gain** — how much including that span reduces the policy's answer-token uncertainty — and uses it as a per-step reward. No oracle evidence labels; the reward comes from the model's own answer distribution.

**Prereqs:** [../_rewards.md](../_rewards.md), [long-cot-rl.md](./long-cot-rl.md), [../grpo.md](../grpo.md)
**Related:** [think-and-read.md](./think-and-read.md) · [../rlvr.md](../rlvr.md) · [prm.md](./prm.md) · [orm.md](./orm.md)

---

## What it is

A reward function for RL on long-context reasoning agents. The agent interleaves **reasoning** with **document consultation** ("think-and-read"); each consulted span `s_k` gets a dense scalar reward measuring the span's utility toward the final answer.

## How it works

**Setup.** The agent produces a trajectory of alternating reasoning turns and read turns. On each read turn `k`, the agent emits a query, retrieves a span `s_k`, and incorporates it before the next reasoning turn.

**Information-gain reward.** For the final-answer span `a`, compare the policy's answer distribution with and without the consulted span:

```
IG(s_k) = KL( π(a | prompt, s_1..k) || π(a | prompt, s_1..k \ s_k) )
```

A span that sharpens the answer distribution (large `IG`) is a useful read. A span that doesn't change the answer distribution (`IG ≈ 0`) is noise. **The reward is per-span, dense across the trajectory**, and anchored to the policy's own uncertainty.

**No evidence labels.** The signal is a function of the policy's output distributions, not of a human-labeled "this span is the answer." It is self-sourced, akin to self-consistency rewards.

**RL compatibility.** LongR plugs into standard RL algorithms — **DAPO** and **GSPO** are both reported — by using `IG(s_k)` as the per-step dense reward and a terminal correctness signal for the final answer.

## Why it matters

- Removes the **evidence-annotation bottleneck** for long-context RL. Standard supervised-evidence pipelines are labor-expensive and don't scale to generic documents.
- Removes the **sparse-reward bottleneck** by giving a dense per-step signal.
- Reported: **+9% on LongBench v2**, consistent improvements on **RULER** and **InfiniteBench**. **Transfers across RL algorithms** (DAPO, GSPO) without re-tuning.
- Distractor-robust: in the paper's analysis, the model learns to assign near-zero IG to irrelevant spans, so the reward resists polluted retrieval corpora.

## Gotchas & tricks

- **`IG` can be gamed by uncertainty inflation.** If the policy artificially widens its answer distribution to make every span look informative, IG becomes meaningless. Pair with a terminal correctness reward (as in the paper) to anchor the policy to right answers.
- **Over-reading.** Dense per-span rewards can encourage reading too many spans. Add a per-read cost or a length penalty — see [length-penalty.md](./length-penalty.md).
- **KL direction.** The direction of KL (predictive-without vs. with) matters for sign and scale; follow the paper's convention and keep the per-sample normalization fixed across the trajectory.
- **Context-window ceiling.** The approach doesn't magically extend a model's effective context — it makes RL *use* the context better. For contexts past the model's native length, pair with position-extension methods.

## Sources

- Paper: *LongR: Unleashing Long-Context Reasoning via Reinforcement Learning with Dense Utility Rewards* — Yu, Chen, Wu, Chen, Yu, Hui, Ping, Yan, Chang — SJTU / PKU / THU / BUPT, 2026 — arXiv:2602.05758.
