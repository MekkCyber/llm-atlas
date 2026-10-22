# Self-Credit
*Depth — credit assignment for long-horizon agent RL by having the policy cite its own evidence and rescaling advantages within a group.*

**TL;DR:** Agent RL with sparse outcome rewards gives every turn in a trajectory the same credit, which blurs out which turns actually mattered. Self-Credit has the policy **write summaries that cite earlier turns** containing the evidence it used; those citations become self-generated per-turn credit signals. An **intra-group rescaling** term then compares cited turns across successful and failed rollouts (same prompt, GRPO-style group), reweighting the advantage so cited turns in successful rollouts get boosted and incidental ones in failed rollouts get demoted.

**Prereqs:** [grpo](../grpo.md), [_rl](../_rl.md), [rlvr](../rlvr.md)
**Related:** [prm](prm.md), [orm](orm.md), [long-cot-rl](long-cot-rl.md), [_rewards](../_rewards.md)

---

## What it is

A credit-assignment layer that sits on top of group-relative RL (e.g. [GRPO](../grpo.md)) for *long-horizon agentic* tasks — tasks where the policy takes many tool-calling turns and only sees reward at the end. The core insight: the policy can emit per-turn credit signals *for free* by referencing which earlier turns' evidence it used.

## How it works

**1. Citation emission.** The system prompt instructs the policy to write a final summary that explicitly cites earlier turns by index (e.g., "based on turns 3 and 7"). The cited turns are flagged inside the trajectory.

**2. Group-relative baseline.** As in GRPO, sample $G$ rollouts per prompt, each receiving a scalar outcome reward $r_i \in \{0, 1\}$ (success/failure).

**3. Intra-group rescaling.** For each turn $t$ in rollout $i$, define a *citation mask* $c_{i,t} \in \{0, 1\}$ (1 if cited in the final summary). Rescale the turn's advantage so that cited turns in successful rollouts get amplified and cited turns in failed rollouts get demoted (and uncited turns are left near the group mean):

$$ \tilde{A}_{i,t} = A_i \cdot \big(1 + \lambda \cdot c_{i,t} \cdot \mathrm{sgn}(r_i - \bar r)\big). $$

The policy-gradient update then uses $\tilde{A}_{i,t}$ in place of the response-wide $A_i$.

**4. Advantage reweighting.** The rescaling raises gradient magnitude on turns that correlate with task success and lowers it on incidental turns, so the policy learns to *reproduce* its own successful evidence-gathering patterns.

## Why it matters

- **Credit assignment without a value head or a learned PRM.** Avoids both the compute of a critic and the hallucination of a learned process-reward model.
- **The signal is self-aligned.** Cited turns are, by construction, the ones the policy thought mattered — reinforcing them has a clean interpretation.
- **Composes with GRPO out of the box.** Only the per-token advantage is modified; the clip/KL/sampling structure is unchanged.

Reports headline gains on long-horizon browsing/deep-search tasks: +37.0 / +36.2 points on BrowseComp-Plus over base Qwen3 8B / 32B, +4.9 on GAIA, +4.7 on xBench-DeepSearch.

## Gotchas & tricks

- **Depends on the policy honestly citing.** If the policy fakes citations to game the credit signal, you get a learned reward-hack. The paper mitigates this by filtering out syntactically-malformed citations.
- **Needs enough variance within a group.** All-success or all-failure groups give zero signal, same as GRPO. Prompt curation to keep solve rates in the 20–80% band is important.
- **Citation format matters.** Freeform `"turns X–Y"` is noisier than a structured JSON schema; the format the policy was trained to emit should be identifiable with a cheap parser.
- **Not a substitute for the base reward.** Self-Credit scales existing advantages; if the base outcome reward is uninformative, the method can't invent a signal.
- **Rescaling coefficient $\lambda$ is the main knob.** Too high and the policy collapses onto whatever it cites most; too low and the credit signal is drowned out. Paper's defaults are a reasonable starting point.
- **Trajectory length should be budgeted.** Very long trajectories dilute the per-turn signal; practical sweet spot is tens of turns.

## Sources

- Paper: *Self-Credit: Credit Assignment for Long-Horizon Agentic RL via Intra-group Rescaling* — Fangcong Yin, Ming Yin et al.; Meta, 2026 — [arXiv:2610.28388](https://arxiv.org/abs/2610.28388).
