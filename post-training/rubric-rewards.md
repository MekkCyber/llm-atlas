# Rubric rewards
*Depth — per-sample, self-evolving rubric criteria as an RL reward signal.*

**TL;DR:** Outcome rewards supervise only the final answer; hand-crafted process rewards use fixed, coarse criteria that saturate as the policy improves. Rubric rewards synthesize a **per-sample rubric** (a set of criteria) grounded in the input, then use those criteria as the reward signal. In AudioRubrics, the rubric is generated from the raw waveform *and* regenerated / reweighted per group conditioned on the current policy's rollouts, so the reward keeps targeting the policy's live weaknesses. Substantially outperforms static-reward RLVR on audio-reasoning benchmarks.

**Prereqs:** [rlvr.md](rlvr.md), [grpo.md](grpo.md), [_rewards.md](_rewards.md)
**Related:** [reasoning/prm.md](reasoning/prm.md), [cot-reward-model.md](cot-reward-model.md), [_rl.md](_rl.md)

---

## What it is

RL post-training's reward function shapes what the policy optimizes for. Two dominant flavors have opposite failure modes:
- **Outcome rewards** (RLVR): let the model reach the right answer without engaging with the input evidence.
- **Fixed process rewards**: hand-crafted rubric criteria that don't adapt per question and don't shift as the policy improves.

Rubric rewards address both by *synthesizing* a rubric per sample from the raw input, then treating each rubric criterion as a scoreable dimension for the RL reward.

## How it works

Per RL step, for each sample $(x, \text{target})$:

1. **Rubric synthesis.** A rubric-generator model $\mathcal{R}$ produces a set of criteria $\{c_1, c_2, \dots\}$ grounded in the input $x$ itself. For AudioRubrics: $\mathcal{R}$ reads the raw waveform and emits criteria like "identifies the primary instrument," "notes the tempo change at 0:12s," etc.
2. **Scoring rollouts against the rubric.** For each rollout $o$, a judge $\mathcal{J}$ scores $o$ against every $c_k$, returning per-criterion scores $s_k(o) \in [0, 1]$.
3. **Composite reward.** Rollout reward $R(o) = \sum_k w_k \cdot s_k(o)$.
4. **Reweight the rubric per group.** After observing rollouts, regenerate / reweight $\{w_k\}$ so that criteria the policy already satisfies deflate and criteria it fails inflate. The reward continuously chases the policy's current weaknesses.
5. **RL update.** GRPO or PPO update with the composite reward.

Because the rubric is *per sample*, criteria are tightly grounded in the specific input rather than being generic templates. Because it's *self-evolving*, static-rubric saturation (a common failure mode of hand-crafted process rewards) is avoided.

## Why it matters

- **Fixes both outcome-reward and static-process-reward failure modes.** Outcome rewards let the model bypass the input; static rubrics saturate. Evolving per-sample rubrics do neither.
- **Substantially better** than open-source and training-based baselines on three audio-reasoning benchmarks (specific numbers in paper).
- **Scale with rubric-generator/judge quality.** Better judge → better reward → better policy. Directly usable as leverage: replace $\mathcal{R}$ or $\mathcal{J}$ with a stronger model to lift the ceiling.
- **Reasoning-length stability.** Policy converges to a stable reasoning length, avoiding both collapse (too short) and unbounded growth (too long) — a bonus that static process rewards don't get for free.

## Gotchas & tricks

- **Judge quality caps everything.** A weak judge produces noisy per-criterion scores; the RL signal degrades.
- **Rubric-generator drift.** If $\mathcal{R}$ is fine-tuned alongside the policy, it can drift and produce rubrics that reward its own biases.
- **Reweighting frequency matters.** Reweight per group (batch) is the paper's default; too frequent → noisy, too rare → static-rubric failure returns.
- **Not applicable everywhere.** Requires an input that can be rubric-decomposed (audio, video, long documents). For math or single-token answers, RLVR is simpler and equivalent.
- **Cost.** Rubric synthesis + per-criterion judging per rollout is more expensive per step than a scalar verifier. Amortize by capping group size.

## Sources

- Paper: *Reinforcement Learning with Evolving Rubrics as Rewards for Audio Reasoning (AudioRubrics)* — Yu, Feng, Min, Lin et al., UMD / UIUC / MSR / MBZUAI, 2026 — arXiv:2608.02831 — https://audiorubrics.github.io/.
