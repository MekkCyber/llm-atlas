# MRPD: Multi-expert Replay Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A method to merge several domain-specific RL-trained expert LLMs into one student *without new rollouts, environment, or reward model*. For each expert, the method **replays the expert's own logged training trajectories** (with their sampling probabilities and rewards, in the order the expert learned from them), the student recomputes the policy gradient on those trajectories, and finally distills the expert's next-token distribution at the logged prefixes. Order of replay matters more than quantity or reward magnitude. Beats both online RL and multi-teacher on-policy distillation on long-horizon agentic benchmarks.

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md)
**Related:** [_rl.md](./_rl.md), [rejection-sampling.md](./rejection-sampling.md), [../case-studies/deepseek-r1.md](../case-studies/deepseek-r1.md), [../case-studies/kimi-k1-5.md](../case-studies/kimi-k1-5.md)

---

## What it is

The "RL per domain, then merge" pattern (reasoning expert + search expert + SQL expert → one generalist) has become standard for frontier reasoning models. Merging is normally done by SFT distillation from online teachers, which is expensive because each teacher must be run fresh on new prompts.

MRPD observes: **the experts already produced the data they were trained on, and those logs contain everything needed to reconstruct their learning signal**. Replay it offline.

---

## How it works

For each expert $k$ with logged training trajectories $\tau^{(k)}_1, \tau^{(k)}_2, \ldots$ (prefix, action, sampling-prob under $\pi_{\text{old}}^{(k)}$, reward), the student $\pi_\theta$ is updated:

1. **Replay in temporal order.** Iterate experts' logs in the order the experts saw them. Interleave across experts or round-robin.
2. **Recompute policy gradient** on the logged tuples with importance-weighted off-policy correction:
   $$
   \nabla_\theta \mathcal{L}_k = -\sum_t \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}^{(k)}(a_t \mid s_t)} \cdot A_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)
   $$
3. **Distillation anchor at prefixes.** At the end of each expert's logs, distill the *final* expert's next-token distribution at a sample of the logged prefixes to anchor the student near each expert's final behavior.

No environment call, no reward model query, no fresh rollouts. The sampling-probability column of the log is what makes off-policy correction well-defined.

---

## Why it matters

- **Dramatically cheaper than online RL merge.** No new rollouts and no reward evaluations. The expensive compute lives in each expert's original training run, which you already paid for.
- **Beats on-policy multi-teacher distillation at a fraction of the cost.** The paper reports on three long-horizon domains (reasoning, search, SQL).
- **Order matters more than amount.** Replaying in the original learning order outperforms uniform random replay, hinting that curriculum signal is preserved in the log sequence.
- **Fits the frontier recipe.** DeepSeek-R1 / Kimi K1.5-style pipelines train several RL experts then merge; MRPD slots in as a cheaper merge step.

---

## Gotchas & tricks

- **Needs full logs.** Just rewards and (prefix, action) pairs aren't enough — you also need the sampling probability under the expert at the time of the rollout. Many open RL frameworks don't log this by default.
- **Importance ratio variance.** When the student has drifted far from $\pi_{\text{old}}^{(k)}$, importance weights blow up. Clipping à la PPO stabilizes.
- **Replay order is a hyperparameter.** Pure round-robin, strict original order, and "mix early-expert late-expert" give different student behaviors; the paper argues original order is best.
- **Doesn't eliminate catastrophic forgetting between experts.** Later-replayed experts can overwrite earlier ones; the distillation anchor is what keeps all of them accessible.

---

## Sources

- Paper: *Replay, Don't Resample: Off-Policy Consolidation of Long-Horizon RL Experts* — Chen, Zeng, Zhuang, Sun, Han, Wang (UIUC / ANL) — 2026 — [arXiv:2611.24059](https://arxiv.org/abs/2611.24059).
- Background: *Off-Policy Policy Gradient* (Degris et al., 2012) — importance-weighted gradient estimation.
- Related: *DeepSeek-R1* — [case-studies/deepseek-r1.md](../case-studies/deepseek-r1.md) — multi-expert → generalist pattern.
