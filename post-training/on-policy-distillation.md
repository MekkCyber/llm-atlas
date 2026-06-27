# On-Policy Skill Distillation

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Adds **dense token-level supervision** to outcome-based agentic RL without external skill memories. Mine "skills" — hindsight summaries of episode-level workflows and step-level decisions — directly from the agent's own completed on-policy trajectories. Re-score the same response with and without the skill injected into the history; the log-probability shift becomes a self-distillation advantage that is added to the outcome advantage. Introduced as **OPID** (2026).

**Prereqs:** [grpo](grpo.md), [_rl](_rl.md), [_rewards](_rewards.md)
**Related:** [rejection-sampling](rejection-sampling.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md), [tool-use-rl-collapse](tool-use-rl-collapse.md)

---

## What it is

Outcome RL (e.g., GRPO over a sparse final-task reward) gives **one scalar per trajectory** in long-horizon agentic tasks. The policy gradient is noisy and silent on *which* intermediate decisions to reinforce. The classic patches — PRMs, intermediate value heads — depend on external labels or heads that don't generalize across tasks.

On-policy skill distillation patches this **without leaving the policy**: every completed trajectory carries hindsight knowledge that, if injected back into the context, makes the same trajectory more probable. Quantifying that lift gives a dense advantage signal.

## How it works

For each completed rollout from prompt $q$ with response $o = (a_1, \dots, a_T)$:

1. **Mine two kinds of skills from the trajectory itself.**
   - **Episode-level skill** — a global workflow summary or failure-avoidance rule extracted from the whole trajectory.
   - **Step-level skill** — a local "what to attend to" description tied to specific critical timesteps.

2. **Critical-first routing.** Identify the critical decision steps in the rollout (e.g., decision-point detector trained or rule-based). At those steps the step-level skill is injected; everywhere else the episode-level skill is the default. This yields, per response, an *augmented history* $q^+ = q \,\Vert\, \text{skill}$.

3. **Re-score with the old policy.** Compute the log-probability of *the same* sampled $o$ under both the original prompt $q$ and the skill-augmented prompt $q^+$ using the **same** old-policy weights $\pi_{\theta_{\text{old}}}$:
   $$
   \Delta_t = \log \pi_{\theta_{\text{old}}}(a_t \mid q^+, a_{<t}) - \log \pi_{\theta_{\text{old}}}(a_t \mid q, a_{<t}).
   $$
   The shift $\Delta_t$ is the **self-distillation token-level advantage** — the policy's own opinion of how much the skill helps each token.

4. **Combine with outcome RL.** The total per-token advantage is
   $$
   A_t = A^{\text{outcome}}(o) + \beta \,\Delta_t,
   $$
   where $A^{\text{outcome}}$ is the group-relative outcome advantage (GRPO-style) and $\beta$ trades off the two signals. Plug into the standard PPO/GRPO clipped objective.

5. **Skill source is the policy itself.** There is no external skill memory, no retrieval store, no privileged context — the skill is harvested per-trajectory from on-policy data.

## Why it matters

- **Distribution-matched supervision.** Earlier skill-distillation recipes use retrieved skills from another policy or dataset; injecting those produces histories the current policy never sees, mismatched with its state distribution. On-policy skills don't.
- **No external infrastructure.** Avoids skill memory stores and retrieval indexes, which were a maintenance and accuracy bottleneck in skill-conditioned agent RL.
- **Keeps RL primary.** The skill term is an *advantage shape*, not a replacement loss. RL stays the optimizer, distillation is the dense shaping.
- **Improves stability and sample efficiency on ALFWorld, WebShop, and Search-QA** in the paper's experiments, over both outcome-only RL and retrieval-based skill-distillation baselines.

## Gotchas & tricks

- **Skill extraction has to be cheap.** A separate LM generates the hindsight skills from trajectories; budget this cost against the RL update budget.
- **Critical-step detection is the easy place to get sloppy.** Naive heuristics over-fire (everything is critical) or under-fire (nothing is). The paper uses an explicit detection step; this is a load-bearing component.
- **$\beta$ matters.** Too high and the policy follows the skill prompt instead of solving the task; too low and the dense signal vanishes.
- **The advantage is computed on the old policy.** Same way GRPO computes the ratio in PPO-style updates — be careful not to mix in current-policy log-probs there, or you reintroduce the high-variance term skill distillation is meant to suppress.
- **Generalizes the "self-improving" pattern.** Conceptually adjacent to STaR / self-distillation pipelines, but realized inline inside the RL update rather than as a separate SFT round.

## Sources

- Paper: *OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning* — Yang, Wu, Lu, Shen, Zhang, Feng, Zhang, Luo, Lian, Wen, Tao, 2026 — [arXiv:2606.26790](https://arxiv.org/abs/2606.26790).
- Background: *DeepSeekMath* — Shao et al., 2024 — origin of GRPO, the outcome-RL backbone OPID layers on top of.
- Background: *STaR: Bootstrapping Reasoning With Reasoning* — Zelikman et al., 2022 — earlier self-distillation-from-own-trajectories idea, applied at SFT time rather than inside RL.
