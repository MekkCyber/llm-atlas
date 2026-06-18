# Winner Advantage Policy Optimization (WAPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A small but principled change to GRPO-style RLVR: **only update on positive-advantage completions**. The paper derives a token-level gradient taxonomy that predicts when each kind of update destabilizes the policy, identifies negative-advantage updates on low-probability tokens as the dominant collapse mode, and shows that dropping them removes the failure without hurting the wins.

**Prereqs:** [grpo](grpo.md), [ppo](ppo.md), [_rl](_rl.md)
**Related:** [rlvr](rlvr.md), [_rewards](_rewards.md)

---

## What it is

GRPO collapse — sudden entropy drop, all rollouts becoming identical, reward hitting a floor — has been one of the most-reported practical issues in reasoning RL. WAPO's diagnosis: when an advantage is *negative* and the token taking the action has *low* probability under the current policy, the PPO-clipped gradient still pushes probability mass off that token, but the mass redistributes onto tokens that were already high-probability, sharpening the distribution and bleeding entropy with every step.

The fix: keep the GRPO machinery — group-relative advantages, PPO-style clipping, KL to a reference — but **discard any rollout with non-positive advantage** before the policy update. Only "winner" trajectories contribute gradient.

---

## How it works

### The token-level taxonomy

For each token in a rollout, the gradient sign and magnitude are determined by two variables:
- **Advantage sign** $\mathrm{sign}(A_i) \in \{+, -\}$
- **Current token probability** $p = \pi_\theta(o_{i,t} \mid \cdots)$, bucketed as low / mid / high

The 6-cell table predicts whether each update will (a) push mass *toward* the right token, (b) raise or lower entropy, and (c) interact safely with the PPO clip. The unsafe cell is $(A<0, p \text{ low})$: the update tries to pull mass *away* from an already-rare token, the clip on the downside fires asymmetrically, and the freed mass lands on high-$p$ tokens — entropy drops without any reward signal justifying it.

### The WAPO objective

Same as GRPO with an indicator gating updates on advantage sign:

$$
L_{\text{WAPO}} = -\frac{1}{G} \sum_{i: A_i > 0} \frac{1}{|o_i|} \sum_t \min\!\left( r_{i,t} A_i,\, \mathrm{clip}(r_{i,t}, 1{-}\epsilon, 1{+}\epsilon) A_i \right) + \beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

where $r_{i,t} = \pi_\theta / \pi_{\theta_{\text{old}}}$ is the per-token PPO ratio and $A_i$ is the group-relative advantage. Negative-advantage rollouts are simply dropped; the KL term still applies to every token.

The group statistics for the baseline are computed over **all** $G$ rollouts (so the mean still calibrates the "winner" cutoff) but only the positive-advantage subset receives gradient.

---

## Why it matters

- **Removes the GRPO collapse mode** observed empirically across reasoning-RL training runs without inventing a new algorithm.
- **No extra cost.** Negative-advantage rollouts are computed (you need them for the group baseline) but skipped at the gradient step — strictly cheaper per backward pass.
- **Grounded explanation, not a hyperparameter.** The taxonomy *predicts* which updates destabilize and the fix follows from it, so practitioners can reason about when WAPO is needed (sparse rewards, many failures per group) vs. when GRPO is fine (high success rate, dense rewards).

---

## Gotchas & tricks

- **All-failure groups give zero gradient.** With WAPO, if every rollout in a group has $A_i \le 0$, the policy doesn't move on that prompt. Combine with prompt-curriculum tricks ([rl-prompt-curation](rl-prompt-curation.md)) or teacher-augmented prompts ([ZPPO](zppo.md)) to keep the winner set non-empty on hard prompts.
- **Group baseline still uses all rollouts.** Don't accidentally drop negative-advantage rollouts before computing the mean / std — the centering would shift and the algorithm would degenerate.
- **Effectively reduces sample efficiency in the easy regime.** When most rollouts succeed (e.g., late-stage RLVR), WAPO and GRPO converge; the gain is concentrated where GRPO is unstable.
- **Compatible with mirror-descent variants.** The same positive-advantage gate can be applied to Kimi-style $\ell_2$-regression updates ([online-policy-mirror-descent](reasoning/online-policy-mirror-descent.md)).

---

## Sources

- Paper: *A Gradient Perspective on RLVR Stability and Winner Advantage Policy Optimization* — Prasanth YSS et al., Layer 6 AI, 2026 — [arXiv:2606.16154](https://arxiv.org/abs/2606.16154).
- Background: *DeepSeekMath* / *DeepSeek-R1* — GRPO's original formulation and the failure modes WAPO addresses.
