# Group Adaptive Clipping Policy Optimization (GAPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** GRPO-family methods use a fixed importance-sampling (IS) clip range $[1-\epsilon, 1+\epsilon]$ for all rollouts, which suppresses exactly the rare-correct rollouts on hard problems that carry the strongest exploration signal. GAPO scales the clip threshold with per-rollout advantage — bigger learning signal buys proportionally more update headroom — motivated by a reverse-KL trust-region view. It is a plug-in replacement for the PPO/GSPO clip; no reward shaping, no new hyperparameter families.

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md), [rlvr.md](./rlvr.md)
**Related:** [_rl.md](./_rl.md), [rl-prompt-curation.md](./rl-prompt-curation.md), [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md)

---

## What it is

RLVR runs with GRPO on hard reasoning prompts spend most of their gradient budget on rollouts near the group mean — because the fixed PPO clip $\epsilon$ applies uniformly. But the *interesting* rollouts are precisely the ones far from the mean: rare correct solutions on hard problems (very high advantage) and near-misses that almost worked (moderate positive advantage). Fixed clipping penalizes the tail as harshly as the middle, and the tail is where new problem-solving behavior lives.

GAPO makes the clip threshold a function of the rollout's advantage $A_i$: rollouts with larger $|A_i|$ get proportionally wider clip range. Formally, replace $\epsilon$ in the PPO surrogate with $\epsilon(A_i) = \epsilon_0 \cdot f(A_i)$ where $f$ is monotone-increasing in $|A_i|$.

## How it works

Start from GRPO's clipped objective:

$$
L = \min\!\left( r_i(\theta) \cdot A_i,\; \mathrm{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) \cdot A_i \right)
$$

with $r_i(\theta) = \pi_\theta(o_i \mid q) / \pi_{\theta_\text{old}}(o_i \mid q)$ the IS ratio. GAPO's change is one line:

$$
\epsilon \;\longrightarrow\; \epsilon(A_i) \;=\; \epsilon_0 \cdot g(|A_i|)
$$

The derivation comes from a reverse-KL trust region: within a KL ball of fixed radius, the maximum allowed IS-ratio deviation is $O(|A_i|)$ at first order, which motivates $g$'s linear-ish shape.

Rollouts with low group success (rare correct on hard problems) have larger raw IS ratios *because* the current policy assigns them low probability; fixed clipping suppresses their gradient contribution to noise level. GAPO reopens that channel exactly for the rollouts whose advantage says they matter.

## Why it matters

- **Small, drop-in change.** GAPO changes only the clip threshold; the PPO/GSPO surrogate, KL penalty, and reward pipeline are untouched. It composes with GRPO, DAPO, GSPO, RLOO.
- **Bigger Pass@k, not just Pass@1.** Because GAPO improves exploration on hard prompts, the largest gains are on Pass@k at the high-k tail — the diversity that most RLVR pipelines silently lose.
- **Targets the regime where you need help.** GAPO's gains are largest when the base model's pass rate is low. On easy prompts where the group mean is already high, GAPO ≈ GRPO — no penalty for adopting it.

## Gotchas & tricks

- **Advantage sign matters.** The scaling should be by $|A_i|$, not $A_i$: negative-advantage rollouts (near-worst) also deserve more downside clip headroom for the same trust-region reason.
- **Interacts with reward normalization.** GAPO reads $|A_i|$; if you z-score across the group, the "hard prompt" signal partially collapses. The paper motivates GAPO after group-relative normalization but before any global reward scaling.
- **Not a substitute for prompt curation.** GAPO widens the tail but does not create new signal — if every rollout in a group is wrong ($\sigma_r = 0$), no clip schedule saves you. Pair with [rl-prompt-curation](./rl-prompt-curation.md).
- **Don't stack with entropy bonuses uncritically.** Both mechanisms buy exploration; using both aggressively can send the policy into a random-search regime.

## Sources

- Paper: *Group Adaptive Clipping Policy Optimization* — Sheng Jia, Xiao Wang, Shiva Prasad Kasiviswanathan, Rein Houthooft — arXiv:2609.00444 (2026).
- Paper: *DeepSeekMath* — Shao et al., 2024 — GRPO baseline.
- Paper: *Proximal Policy Optimization* — Schulman et al., 2017 — origin of the clipped IS surrogate.
