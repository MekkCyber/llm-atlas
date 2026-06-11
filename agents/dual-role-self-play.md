# Dual-role self-play

*Depth — one LLM playing both the agent and the environment for agent training.*

**TL;DR:** Agent RL needs three things: an environment, a reward, and a way to assign credit. Role-Agent (2026) folds all three into the same LLM. As **World-In-Agent (WIA)** it predicts the environment's next state; the prediction error becomes a dense process reward. As **Agent-In-World (AIW)** it analyzes its own failure rollouts and retrieves similar past failures to reshape the training distribution. The same backbone plays both roles during training; +4% average over strong agent baselines.

**Prereqs:** [grpo](../post-training/grpo.md), [_rl](../post-training/_rl.md)
**Related:** [prm](../post-training/reasoning/prm.md) · [harness-optimization](harness-optimization.md) · [delegation-sft](delegation-sft.md)

---

## What it is

For long-horizon agent tasks, sparse task-level rewards make credit assignment hopeless and out-of-distribution failures dominate evaluation. The usual fixes are expensive: train a separate world model, train a separate process reward model, hand-author harder tasks.

Dual-role self-play has one LLM cover all three jobs:

- The **environment role (WIA)** is the same LLM asked to predict $s_{t+1}$ given $(s_t, a_t)$. Its prediction error becomes a step-level signal that the *agent* can use as a process reward.
- The **agent role** is the same LLM as a policy producing actions.
- The **data-engineer role (AIW)** is the same LLM asked to look at failure trajectories and produce a retrieval prompt → training distribution edit.

Same weights, different prompts. The "self-play" framing is closer to LLM-as-judge than to AlphaZero — the LLM plays multiple roles over its own rollouts.

## How it works

### World-In-Agent (process reward from next-state prediction)

At each step $t$ during rollout:
1. The agent executes action $a_t$, environment returns true next state $s_{t+1}$.
2. The same LLM (WIA role, different prompt) predicts $\hat{s}_{t+1}$ from $(s_t, a_t)$.
3. Process reward at step $t$:
   $$ r^\text{proc}_t = -\| \hat{s}_{t+1} - s_{t+1} \|^2 \quad \text{(or token-level similarity)} $$
4. Combine with sparse task reward: $R_t = r^\text{task} + \lambda \cdot r^\text{proc}_t$.

The intuition: if the agent *understands the environment well enough to predict it*, that capability correlates with task-relevant action selection. Reward correct prediction and you reward calibration — which helps credit assignment.

### Agent-In-World (failure-driven data reshaping)

After a batch of rollouts:
1. Cluster failures.
2. For each cluster, prompt the LLM (AIW role) to retrieve similar tasks from a task bank.
3. The retrieved tasks reshape the next training-batch sampling distribution toward the model's actual failure modes.

This is a meta-loop on top of the main RL — analogous to curriculum learning but driven by the model's own failure analysis instead of a fixed schedule.

### Putting them together

WIA gives dense per-step gradient; AIW makes sure the gradient is spent on hard tasks. Both loops run continuously during training.

## Why it matters

- **No separately-trained world or reward model.** Both signals come from the same backbone. Cheaper and stays in-distribution as the agent improves.
- **Step-level process reward without PRM labels.** WIA's next-state prediction is a self-supervised target — no human-annotated step quality required.
- **Generalizes to any agent task with an observable state**, which is most of them.
- **Composes with GRPO/PPO.** WIA is just an extra reward term; AIW is just a sampling distribution. Drop-in.

## Gotchas & tricks

- **WIA needs an observable state.** Tasks where $s_{t+1}$ is hidden or partially observable break the next-state-prediction signal. Workaround: predict the *observation* from the environment, not the latent state.
- **Process reward can swamp task reward.** Calibrate $\lambda$ — too high and the agent optimizes for predictable trajectories rather than successful ones.
- **AIW's "similar task" retrieval can over-concentrate.** If similar failures pull from a narrow slice of the task bank, the agent over-trains on one failure mode. Diversity-aware retrieval mitigates.
- **Prompt boundaries matter.** Sharing weights across roles only works if the prompts are clearly distinct — otherwise the model conflates "predict environment" with "act" and degrades both.
- **Doesn't replace exploration bonuses.** WIA rewards calibration, not novelty. If the agent's policy is stuck, WIA won't break it loose.

## Sources

- Paper: *Role-Agent: Bootstrapping LLM Agents via Dual-Role Evolution* — Wang et al., USTC / AMAP Alibaba, 2026 — [arXiv 2606.10917](https://arxiv.org/abs/2606.10917).
- Background: process reward models — [prm](../post-training/reasoning/prm.md).
