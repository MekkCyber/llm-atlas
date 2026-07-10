# Single-Rollout Asynchronous Optimization (SAO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** GRPO-style **group-wise sampling** ($G$ rollouts per prompt, group-mean baseline) does not fit **asynchronous** long-horizon agentic RL: rollouts take minutes-to-hours, workers need to update the model as trajectories arrive, and forcing a group barrier wastes GPU. SAO takes **one rollout per prompt**, restores a lightweight learned **value network** for the baseline, and adds a **strict double-sided token-level clip** to keep off-policy updates stable. Introduced by Tsinghua and used as the RL post-training pipeline for GLM-5.2 (750B-A40B).

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [rlvr.md](./rlvr.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Asynchronous RL for LLMs updates the policy as rollouts arrive from a fleet of workers, instead of waiting for a full synchronous batch. It's the natural fit for **long-horizon agent tasks** (SWE-Bench, IMO-style theorem proving, tool-use trajectories) where a single rollout can run for tens of minutes.

GRPO — the dominant modern RL algorithm for LLMs — needs $G$ rollouts *from the same prompt* to compute its group-mean baseline. Under asynchrony, this becomes a barrier: the update for prompt $q$ waits for the slowest of its $G$ rollouts. On agentic tasks where rollout time varies by 10×, the group barrier is the bottleneck.

SAO drops the group-sampling assumption and re-solves the resulting baseline problem with a learned value net + a tighter clip.

## How it works

**One rollout per prompt.** For each prompt $q$, sample **one** trajectory $o$ from $\pi_{\theta_{\text{old}}}$ and compute its terminal reward $r$. This trivially satisfies async: prompt $q$'s update fires as soon as $o$ completes, no barrier.

**Value network baseline.** Because the group-mean baseline is gone, SAO trains a small value network $V_\phi(s)$ jointly with the policy. Advantage:
$$A_t = R_t - V_\phi(s_t)$$
$V_\phi$ is trained with a plain MSE loss against returns-to-go. The paper argues that with proper warmup and off-policy correction, a value net at LLM scale is not the burden that GRPO papers claimed.

**Strict double-sided token-level clip.** Async introduces more off-policy skew than sync PPO. SAO applies a *double-sided* clip: the standard PPO ratio clip **and** a hard cap on $|A_t|$ per token before entering the PPO minimum. Both directions of the clip fire, so a large-magnitude off-policy advantage cannot dominate a step.

The full objective is otherwise PPO: KL to a reference $\pi_{\text{ref}}$, GAE-style value targets, standard entropy regularization.

## Why it matters

- **Unblocks async agentic RL.** Group-wise sampling was the reason async pipelines had to synthesize fake barriers or run wasteful padded rollouts. Single-rollout removes both.
- **The value network is back.** The GRPO era treated the value net as dead weight; SAO shows it's exactly what async long-horizon RL needs — a stable, prompt-agnostic baseline that doesn't depend on group completeness.
- **Trains stably for 1,000+ async steps** where GRPO variants diverge; beats GRPO and DAPO on SWE-Bench Verified, BeyondAIME, and IMOAnswerBench.
- **Deployed at scale.** Powers the RL post-training pipeline for the open **GLM-5.2 (750B / 40B active)** release — a real-world artifact showing async single-rollout RL works at frontier scale.
- **Fits online learning.** SAO is particularly effective when the environment shifts during training (evolving toolchains, changing tests) — single-rollout adapts step-by-step, GRPO's group-mean lags.

## Gotchas & tricks

- **Value net warmup matters.** Cold value net produces zero-mean advantages and no policy signal for the first ~50 steps. Warm from a small supervised regression on frozen-policy rollouts.
- **Clip both sides.** Author's ablation: dropping either half of the double-sided clip re-introduces divergence within 200 steps.
- **Off-policy staleness.** Even with one rollout per prompt, the trajectory was sampled from a stale $\pi_{\theta_{\text{old}}}$ that may be many updates behind. The value net absorbs some of this; the tight clip absorbs the rest.
- **Reward scaling.** With no group std normalization, reward magnitudes matter globally. Add a running mean-std normalization on $r$ across the batch.
- **Not a free win on RLVR math.** On short-horizon verifiable tasks with cheap parallel rollouts, GRPO's group baseline is still competitive — SAO's win is concentrated in long-horizon agent settings.

## Sources

- Paper: *Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning* — Hou, Li, Tang, Dong (Tsinghua), 2026 — arXiv:2607.07508.
- Related: *DeepSeekMath (GRPO)* — Shao et al., 2024 — the group-sampling baseline SAO replaces.
- Deployment: *GLM-5.2 technical release* — Z.ai / Tsinghua — SAO is the RL stage.
