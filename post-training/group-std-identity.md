# Group-Standard-Deviation Identity
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Under **binary verifiable rewards** (right/wrong), the standard deviation of a GRPO group's rewards is *exactly* determined by how many of the $G$ rollouts are correct — and it is *exactly* the size of the training update. GRPO divides by $\sigma_r$, Dr. GRPO drops the division, DAPO discards $\sigma_r = 0$ groups; all three are three settings of one dial. A **split** group ($k \approx G/2$) teaches the most; a **unanimous** group ($k \in \{0, G\}$) teaches nothing.

**Prereqs:** [grpo](grpo.md), [rlvr](rlvr.md), [_rl](_rl.md)
**Related:** [rl-prompt-curation](rl-prompt-curation.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

For a group of $G$ rollouts scored with a binary verifier $r_i \in \{0, 1\}$, let $k = \sum_i r_i$ be the number of correct answers. Then

$$
\bar{r} = \tfrac{k}{G}, \qquad \sigma_r = \sqrt{\tfrac{k}{G}\left(1 - \tfrac{k}{G}\right)}
$$

so $\sigma_r$ is a smooth function of $k$ alone — maximal at $k = G/2$, zero at $k \in \{0, G\}$. Under the standard GRPO advantage $A_i = (r_i - \bar{r})/\sigma_r$, the gradient's contribution from that group is a monotone function of $\sigma_r$. This is the **group-standard-deviation identity**: *update magnitude equals group disagreement*.

The identity re-reads the three most-cited GRPO variants as three settings of one dial:

- **GRPO** divides by $\sigma_r$ → normalises update magnitude across prompts.
- **Dr. GRPO** drops the $\sigma_r$ division → keeps raw $r_i - \bar{r}$, so easier prompts (large $\sigma_r$) update more.
- **DAPO** discards groups with $\sigma_r = 0$ → skips unanimous groups instead of dividing by zero.

## How it works

The proof is short. For binary rewards, $r_i^2 = r_i$ so $\mathrm{Var}(r) = \bar{r} - \bar{r}^2 = \bar{r}(1-\bar{r})$. Substituting $\bar{r} = k/G$ gives the closed form for $\sigma_r$. Plug into the PPO-clipped GRPO objective and the per-group gradient contribution simplifies to

$$
\|\nabla L_{\text{GRPO}}\|_{\text{group}} \;\propto\; \sigma_r \cdot \|\nabla \log \pi_\theta\|
$$

up to per-token clipping — an *exact* proportionality between disagreement and update size. The choice of normalisation (divide, drop, or gate) selects **where in difficulty-space** the learner spends its gradient budget:

- Divide (**GRPO**): equalises across prompts; hard-and-easy prompts contribute similar update magnitudes.
- No divide (**Dr. GRPO**): near-50/50 prompts dominate — the split-difficulty regime learns most.
- Gate (**DAPO**): filter out the exactly-degenerate cases and keep everything else on either policy.

## Why it matters

The 2025 reasoning-RL literature spent significant effort proposing "fixes" to GRPO's normalisation. The identity says those fixes are the same knob relabelled — the algorithmic surface is thinner than the paper count suggests, and the useful lever is elsewhere.

The right lever is **prompt-difficulty curation**: pick prompts whose sampled $\sigma_r$ is high, not whose accuracy is high. This is a much sharper target than "hard problems" — a 100% wrong problem contributes nothing under any of the three algorithms.

## Gotchas & tricks

- The identity holds *exactly* only under binary rewards. Composite rewards (accuracy + format + language, as in R1) blunt it — $\sigma_r$ is no longer a function of $k$ alone.
- Larger $G$ makes the $\sigma_r = 0$ tail rarer but never zero — DAPO's gate matters most at small $G$ (e.g. $G = 4$).
- Under the identity, a difficulty-curated batch with $\sigma_r > 0$ everywhere makes the GRPO-vs-Dr. GRPO distinction *empirically vanish* — the two only diverge when the batch mixes hard-unanimous with easy-split prompts.
- If the verifier is noisy (probabilistic RM instead of a rule), the closed form breaks; use the empirical $\sigma_r$ but don't expect the identity's monotonicity.

## Sources

- Paper: *GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number: The Group-Standard-Deviation Identity* — Bay & Yearick, 2026 — [arXiv:2607.00152](https://arxiv.org/abs/2607.00152).
- Paper: *DeepSeekMath* — Shao et al., 2024 — introduces GRPO with the $\sigma_r$ division.
- Paper: *DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization* — 2024/25 — introduces the $\sigma_r = 0$ gate.
