# Domain-Specialised RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of running one RL loop over a batched mixture of all capabilities, **domain-specialised RL** splits the policy's post-training into **per-capability RL loops** — each with its own reward model, its own roll-out environment, and its own hyperparameters. Signals don't average across capability modes (which blurs them), and each cluster gets a tuned recipe. Used in IQuest-Q1's post-training to strengthen agentic-coding capability without washing out general reasoning (IQuest, 2026).

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md)
**Related:** [rlvr.md](rlvr.md), [_rewards.md](_rewards.md), [../case-studies/iquest-q1.md](../case-studies/iquest-q1.md)

---

## What it is

A factorisation of RL post-training by capability cluster. For an agent that must do code editing, tool use, long-horizon reasoning, retrieval, and general chat, domain-specialised RL defines a reward + environment per cluster and runs each loop with the tuning each needs — rather than relying on a single global reward to balance them.

## How it works

- **Cluster definition.** Decide the capability clusters (e.g. coding, agentic tool use, math reasoning, writing). Each must have an evaluable success criterion.
- **Per-cluster reward.** Each cluster has its own reward model or verifier. Coding uses unit-test pass; tool-use uses environment success / step reward; math uses answer-match verifier; writing uses a preference model.
- **Per-cluster roll-out environment.** Code rollouts live in a sandbox; tool-use rollouts live in an agent harness; math rollouts live in a straight-shot generator. Mixing these in one data loader is painful; dedicating a loop per cluster keeps the stack simple.
- **Scheduler.** Loops are interleaved (round-robin or weighted by cluster weakness). Checkpoints flow between loops; the policy at the end of cluster A's step becomes the init for cluster B's next step.
- **Guard against regression.** After each loop, evaluate on *other* clusters' holdouts. If capability regresses, back off or reweight.

## Why it matters

- **Signals don't blur.** A single-reward mix dilutes clusters where the reward is sparse (coding: binary pass/fail) with clusters where it's dense (preference rewards). Per-cluster loops preserve signal fidelity.
- **Tuned recipes per cluster.** Coding RL wants large G (many rollouts per prompt) and strict verifiers; tool-use RL wants long-horizon environments and partial-credit rewards; math wants rule-based verifiers. One global recipe is a compromise for all; per-cluster recipes aren't.
- **Compounds with MOPD.** Specialists trained with domain-specialised RL become the teachers for MOPD. Together they form the IQuest-Q1 post-training backbone.

## Gotchas & tricks

- **Cluster regression is the main failure mode.** Pouring RL into coding can measurably reduce writing / general-reasoning scores if guard rails aren't tight. Cross-cluster holdouts are mandatory.
- **Reward hacking per cluster.** Each cluster's reward is narrow; the policy can learn to game it in cluster A at the cost of cluster B. KL to a shared reference model helps.
- **Scheduler design is empirical.** Round-robin, hardest-first, and weakness-weighted all have proponents; no universal winner.
- **Capability-cluster boundaries are fuzzy.** "Reasoning" overlaps with coding and math; drawing clean cluster lines matters more than it should.
- **Not a replacement for a single mixed loop at the end.** Many recipes end with a global polish loop over the mixed distribution to re-balance; IQuest-Q1's "multi-scaffold multi-version" wording suggests something similar.

## Sources

- Paper: *IQuest-Q1: Advancing Foundation Capabilities for Agentic CLI Systems* — IQuest, 2026 — [arXiv:2610.30949](https://arxiv.org/abs/2610.30949) — introduces domain-specialised RL as a named ingredient of its post-training.
