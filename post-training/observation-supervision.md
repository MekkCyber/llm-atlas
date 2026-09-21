# Observation Supervision (ActObs)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard agent SFT applies the loss only to agent-authored action tokens and treats environment observations as context. Observation supervision (ActObs) also **supervises the observation tokens** — the student learns to predict what the environment will say next, even though deployed agents never generate observations. The change is free: same data, same parameters, no extra forward passes. After SFT the two initializations look similar; after GRPO, ActObs consistently reaches higher pass@k with more entropy retained and less policy drift.

**Prereqs:** [grpo.md](./grpo.md), [rejection-sampling.md](./rejection-sampling.md)
**Related:** [rlvr.md](./rlvr.md) · [_rl.md](./_rl.md) · [../agents/README.md](../agents/README.md)

---

## What it is

A trajectory recorded by an agent is a sequence of alternating actions and observations: the agent emits `a_t`, the environment returns `o_t`, the agent emits `a_{t+1}`. Action-only SFT masks the observation tokens out of the loss — they're inputs, not targets.

ActObs removes that mask. The observation tokens already sit in the sequence; supervising them costs nothing but a change to the loss mask. What it buys is that the policy is now trained to model **action consequences** — a next-state model, learned for free alongside next-action prediction.

## How it works

Given a trajectory $\tau = (q, a_1, o_1, a_2, o_2, \ldots)$:

- **Action-only SFT (baseline):** $L_{\text{SFT}} = -\sum_{t} \log \pi_\theta(a_t \mid q, \tau_{<t})$ where the sum is over action tokens.
- **ActObs:** $L_{\text{ActObs}} = -\sum_{t} \log \pi_\theta(a_t \mid \ldots) - \lambda \sum_{t} \log \pi_\theta(o_t \mid \ldots)$ where $\lambda$ balances the two. The observation gradient flows into the same shared parameters as the action gradient.

At inference the policy still only *emits* actions — the observation-supervision head is unused. It shaped the parameters during training; it doesn't change deployment.

## Why it matters

- **Free change.** No new data, no new parameters, no extra forward/backward passes. The observation tokens were already in the SFT sequence.
- **Better RL initialization.** GRPO from ActObs SFT beats GRPO from action-only SFT at every sampling budget on Terminal-Bench 2.0 with Qwen3-4B. On Qwen3-8B it trades some pass@1 for higher pass@16 (+3.4 pp). On aider-polyglot (out-of-domain), +4.2 pp pass@1 at 4B.
- **More entropy, less drift.** ActObs policies retain more entropy through RL and stay closer to the SFT initialization — exploration remains open longer.
- **Prevents the specialization pathology.** In action-only SFT, action and observation gradients rapidly become orthogonal and the model actively *degrades* below the base model at environment prediction. Joint supervision prevents that one-sided specialization.

## Gotchas & tricks

- **$\lambda$ matters at the boundary.** Too low: nothing changes vs. action-only. Too high: the policy over-weights observation modeling and action likelihood degrades. Start $\lambda = 1$ and sweep down.
- **Observation content dominates loss.** In many environments, observations are longer than actions and full of routine boilerplate (e.g., prompts from the shell, dump listings). Consider per-token normalization or downweighting tokens that carry no signal.
- **Not RL-time.** ActObs is an SFT-stage intervention. During the RL rollout, only actions are sampled from the policy and observations come from the environment as usual.
- **Effect on pass@1 vs pass@k.** ActObs biases the policy toward exploration (higher entropy). Expect stronger gains on pass@k than pass@1; sometimes small pass@1 regressions.
- **Downstream code-editing bonus.** Cross-domain transfer to unseen tasks (aider-polyglot after Terminal-Bench 2.0 SFT/RL) suggests the environment-prediction skill generalizes beyond the training environment.

## Sources

- Paper: *Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL* — Zhang et al., Amazon, 2026 — arXiv:2609.20715.
