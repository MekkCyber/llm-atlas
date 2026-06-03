# ESPO — Early-Stopping Proximal Policy Optimization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A drop-in PPO/GRPO/DAPO modifier for long-CoT RL that **detects irrecoverable trajectory failure on the fly** and terminates the rollout, mapping the truncation to an absorbing failure state with a terminal penalty. Uses only signals already computed by a standard PPO step — the policy's logit gap (surrogate regret) and the critic's value estimate — so no extra reward model or annotations. On DeepSeek-R1-Distill-Qwen-7B, ESPO surpasses PPO on AIME 2024 (46.28% vs 45.25%), AMC 2023 (85.83% vs 82.94%), and MATH-500 (87.42% vs 85.43%), while saving >20% rollout tokens.

**Prereqs:** [../ppo.md](../ppo.md), [../grpo.md](../grpo.md), [_rl](../_rl.md)
**Related:** [long-cot-rl](long-cot-rl.md), [length-penalty](length-penalty.md), [../rlvr.md](../rlvr.md)

---

## What it is

In long-CoT RL, when a model commits to a wrong reasoning step early, standard algorithms force generation to continue until $T_{max}$. The post-failure tokens earn no reward but still enter the advantage estimate — adding noise gradients that misdirect learning *and* burning rollout compute. ESPO detects failure during generation and truncates, isolating the noisy post-failure region from the PPO update.

ESPO modifies rollout *collection*, not the PPO objective itself: PPO and GAE run unchanged on the truncated trajectories.

---

## How it works

### Per-step deviation signal

At decode step $t$, with sampled token $a_t \sim \pi_\theta(\cdot|s_t)$:

$$g_t = \max_{a \in \mathcal{V}} \log\pi_\theta(a|s_t) - \log\pi_\theta(a_t|s_t)$$

$g_t \geq 0$ — small when the sampled token is near the policy mode, large when sampling deviates from the mode (a proxy for being in a "lost" state). Free: $g_t$ is computed from logits the decoder already produced.

### Normalized cumulative regret

EMA-normalize $g_t$ (batch statistics frozen during rollout for causal correctness):

$$\tilde g_t = \mathrm{clip}\!\left(\tfrac{g_t - \mu_g}{\sqrt{\sigma_g^2 + \delta}}, -c, c\right)$$

Accumulate within the trajectory with smoothing $\alpha_s$:

$$z_t = \alpha_s z_{t-1} + (1-\alpha_s)\tilde g_t$$

### Value-gated stopping rule

Terminate at step $t$ if:

$$z_t > \beta \cdot \max(V_\phi(s_t), \varepsilon)$$

Interpretation: states the critic still expects value from get more tolerance; low-value states are cut after a smaller accumulated regret. A proportional controller adjusts $\beta$ to track a target termination rate (default 25%).

### Failure transition

Truncation is mapped to an absorbing failure transition with $r_{T_{stop}} = r_{fail}$ (default $-1$). GAE propagates this concentrated negative TD-error back to earlier tokens. No bootstrap beyond the absorbing state. Crucially, this is *not* a per-step length penalty — those create non-stationary rewards that collapse the policy's logit spread.

### Critic warmup

Disable the stopping rule during a warmup phase (until critic loss stabilizes — adaptive, capped at 10% of training steps). A randomly initialized critic produces uncalibrated value baselines that would trigger spurious truncations.

---

## Why it matters

- **Improves accuracy *and* cuts compute simultaneously.** On 7B: +1.97 pp average over PPO and +1.41 pp over DAPO, with 22% fewer rollout tokens than PPO. Random-truncation at the same rate scores 42.4 vs ESPO's 46.3 on AIME24 — the gain comes from *where* trajectories are cut.
- **Orthogonal to advantage-estimation improvements.** PPO/GRPO/DAPO all tackle the variance / credit-assignment problem on advantages; ESPO removes a different noise source (post-failure tokens) before advantages are computed. Composes with all of them.
- **No extra reward model / annotations.** Earlier "stop early when failed" approaches needed process reward models (PRMs) or learned termination heads (Option-Critic). ESPO reuses the actor's logits and the critic's value — already in the PPO forward pass.
- **Slows entropy collapse.** By removing the misattributed negative-gradient pressure on post-failure tokens, ESPO preserves the policy's distributional spread better than vanilla PPO.

---

## Gotchas & tricks

- **The critic warmup matters.** Skipping it (variant B in ablations) costs 2.1 pp on AIME24 because spurious early terminations swamp the signal before the value baseline calibrates.
- **The terminal penalty is the right shape.** Removing it (variant C) costs 2.6 pp — a concentrated TD-error at the truncation point gives clean credit assignment; spread penalties don't.
- **Value-gating combined with regret-gating beats either alone.** Value-only stop (D): 44.0; regret-only (E): 44.8; combined (A): 46.3. Value alone depends on absolute critic scale; regret alone lacks recovery allowance.
- **Confidently-wrong is the failure mode.** When the policy is confidently incorrect, $g_t \approx 0$ and ESPO under-truncates. The paper flags this; future work could add a value-anomaly term.
- **False-positive rate is small.** ~2.7% of trajectories that *would* have recovered get cut. Net positive but not free.
- **Currently tied to PPO/DAPO** (actor-critic). GRPO with no critic needs a different value surrogate to plug in.

---

## Sources

- Paper: *ESPO: Early-Stopping Proximal Policy Optimization* — Li, Zhou, Shi, Yu, Tan, Liu, Li, Li, Li, Yang, Ye — Tongyi Lab / Alibaba & Peking University, 2026 — [arXiv:2605.29860](https://arxiv.org/abs/2605.29860).
- Background: *Time Limits in Reinforcement Learning* — Pardo et al., ICML 2018 — the absorbing-state framing for truncated rollouts.
- Background: *DAPO: An Open-Source LLM RL System at Scale* — Yu et al., 2025 — the DAPO baseline ESPO improves over.
