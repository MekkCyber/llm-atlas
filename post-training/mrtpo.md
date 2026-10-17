# MRTPO (Multi-Reward Token-level Policy Optimization)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** When training against several verifiable rewards at once (correctness + format + per-constraint, or SQL schema + semantics + output), reward-decoupled normalization (GDPO) keeps each reward's advantage distinct but still spreads it evenly over every token in the response. **MRTPO** attributes each reward's advantage to the tokens that actually depend on it, using a **Jensen–Shannon replay** — compare the next-token distribution with vs without each verifier's reference information to find per-token relevance. No auxiliary attribution model, no token-level annotations.

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md), [_rewards.md](./_rewards.md), [_rl.md](./_rl.md)
**Related:** [rl-prompt-curation.md](./rl-prompt-curation.md)

---

## What it is

In multi-reward RL you have verifiers $v_1, \ldots, v_M$ each returning a scalar per response. The dominant approach is to compute per-reward advantages $A_i^{(m)}$ (GDPO-style) and sum them, broadcasting each $A_i^{(m)}$ to **every token** of response $o_i$. The flaw: a format verifier only cares about tags; a SQL-semantics verifier only cares about the FROM/JOIN region. Rewarding every token for a format fix teaches the model that any token could be the fix.

MRTPO measures *which tokens the verifier $v_m$ actually looks at* using the policy itself, no new model.

## How it works

For each response $o_i$ and each reward $v_m$:

1. **Define the verifier's reference info** $c_m$: the ground-truth answer for a correctness verifier, the required format spec for a format verifier, the schema for a SQL verifier. This is already available at train time.
2. **Replay with and without $c_m$.** Run the policy on the prompt $q$ and prefix $o_{i,<t}$ twice:
   - with $c_m$ appended to the context (visible): $p_t^{\text{with}} = \pi(\cdot \mid q, c_m, o_{i,<t})$
   - without $c_m$ (hidden): $p_t^{\text{without}} = \pi(\cdot \mid q, o_{i,<t})$
3. **Per-token relevance** = Jensen–Shannon distance:
   $$ \alpha_{i,m,t} = \mathrm{JS}(p_t^{\text{with}} \,\|\, p_t^{\text{without}}) $$
   Normalize across $t$ so $\sum_t \alpha_{i,m,t} = 1$ per response.
4. **Reassign advantage.** Instead of broadcasting $A_i^{(m)}$ uniformly over tokens, assign per-token advantage
   $$ A_{i,m,t} = \alpha_{i,m,t} \cdot L \cdot A_i^{(m)} $$
   where $L$ is the response length. Total advantage mass per response is preserved; distribution is concentrated.
5. **Sum across rewards and run GRPO.** Final per-token advantage $A_{i,t} = \sum_m A_{i,m,t}$; otherwise standard GRPO+clip.

## Why it matters

- **Uses signal the verifier already carries.** The reference info $c_m$ was the ground truth to grade the response; MRTPO repurposes it as a *token-level attribution probe* for free.
- **Faster convergence, better final scores.** Reported on text-to-SQL, multi-constraint instruction following, and tool calling vs reward-decoupled baselines.
- **No new network, no labels.** Compared to learned token-level critics or expensive labeled attributions, this is almost free (two forward passes per response per reward).

## Gotchas & tricks

- **$c_m$ must actually be informative.** For a free-form helpfulness reward with no reference, there is no $c_m$ to ablate — MRTPO falls back to uniform (and so to GDPO) for that reward.
- **Two forward passes per reward per response.** Not free: if $M$ is large or rollouts are long, this scales. Paper evaluates with $M \le 3$–5.
- **Order of operations.** Compute $\alpha$ **before** the policy update step — it's a property of the *rollout-time* policy, not the updating one.
- **Doesn't fix mis-specified rewards.** If a verifier is wrong, attributing credit more sharply makes the mistake sharper too. MRTPO improves credit assignment, not reward design.
- **Compatible with GRPO advantages.** The $A_i^{(m)}$ themselves can be computed with any GRPO/RLVR variant — MRTPO is a reweighting layer, not a new objective.

## Sources

- Paper: *Each Reward to Its Relevant Tokens: Reward-Specific Token-Level Credit Assignment for Multi-Reward RL* — Wang (NUS), Liu (UIUC), Song (USC), Ding (UPenn), Liang (UMN), Kang (MBZUAI), 2026 — [arXiv:2610.21439](https://arxiv.org/abs/2610.21439).
- Code: https://github.com/zhaihaotian/MRTPO
- Baseline: reward-decoupled normalization (GDPO) and standard summed-reward GRPO/RLVR.
