# CARE — Competence-Aware Reward-and-Advantage Engineering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A cost-aware reward layer for online agentic RL. CARE reshapes the reward and advantage of a group-relative RL objective so the policy is pushed to *reduce reasoning tokens and tool calls* while preserving task success. Introduced in Qwen-Planner-Agent for training long-horizon mobile-app agents where real-device rollouts are expensive.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/_rewards.md](../post-training/_rewards.md)
**Related:** [README.md](README.md), [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

Standard agentic RL rewards outcome only ("did the workflow succeed?"). That trains a policy that *does* succeed but may burn tokens and tool calls doing so — a real problem when each rollout hits a physical device or an expensive tool.

CARE adds a *competence-aware* cost term: reward the shortest path to success, penalize wasted effort. Crucially it does this at the **advantage** level, not just the raw reward, so the RL signal accounts for the current policy's competence when deciding how much to punish inefficiency.

## How it works

Within a GRPO-style group of `G` rollouts for the same prompt:

- Compute a base outcome reward `r_task` (1 if success, 0 otherwise, or a graded score).
- Compute an **effort penalty** proportional to reasoning tokens plus tool calls used.
- Blend into a shaped reward that rewards success **and** parsimony.
- At the advantage step, scale the penalty by a *competence signal* — the policy's current success rate on the prompt. When success is rare, back off on the efficiency penalty (learning to *succeed at all* comes first); when success is reliable, ramp up the penalty (now compress the trajectory).

The RL update is otherwise standard GRPO — group-relative baseline, clipped ratio, KL to reference.

## Why it matters

- Directly addresses "why is this agent so verbose?" by making concision a first-class objective.
- The competence gating avoids the classic failure mode where cost penalties kick in before the policy has learned to succeed at all.
- Transferable beyond mobile: any high-latency environment (computer use, browser use, robotics) has the same efficiency-vs-success tension.

## Gotchas & tricks

- The competence estimator is per-prompt and needs enough rollouts to be stable; small `G` hurts.
- Effort penalties should be normalized within the group; absolute-token-count penalties are unfair across prompts of different difficulty.
- Reward hacking: the policy can learn to skip *useful* reasoning to save tokens. Combine CARE with rubric-style intermediate rewards or process rewards if final-outcome-only signal proves too coarse.
- Not a replacement for verifiable-reward RL — CARE is a shaping layer *on top* of a base reward, not the whole thing.

## Sources

- Paper: *Qwen-Planner-Agent: A Closed-Loop AI-for-AI Framework for Real-World Mobile Planner Agents* — Qu, Sun, Liu, Zhao, Ding, Wang, Cao, Jiao, Xie, Wu, Wang, Zhang, Liu, Wang, Xie, Zhao, Chen, Ma, Yang, Zhao, Jia, Gong, Zhu, Zhong, Hoi and more — 2026 — [arXiv:2609.29892](https://arxiv.org/abs/2609.29892).
