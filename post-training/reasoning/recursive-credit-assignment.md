# Recursive Turn-Level Credit Assignment

*Depth — a critic-free per-turn credit signal for agentic RL, computed from token-level teacher–student log-prob gaps and updated as a Bayesian log-odds belief.*

**TL;DR:** Trajectory-level RL for LLM agents (GRPO + verifiable outcome rewards) doesn't tell you which of the N tool calls in a rollout actually mattered. Recursive turn-level credit assignment (introduced by AgentOPSD, 2026) fixes this without training a critic and without extra rollouts: aggregate token-level teacher–student log-prob gaps into per-turn "evidence," update a Bayesian belief over turn-pivotalness in log-odds space, and reweight the trajectory advantage by the marginal belief revision between consecutive turns. Drops in on top of GRPO/PPO.

**Prereqs:** [../grpo.md](../grpo.md), [../rlvr.md](../rlvr.md)
**Related:** [orm.md](orm.md), [prm.md](prm.md), [../rejection-sampling.md](../rejection-sampling.md), [long-cot-rl.md](long-cot-rl.md)

---

## What it is

An **advantage-reweighting** scheme for multi-turn agentic RL. Instead of assigning a single trajectory advantage `A` uniformly to every token in every turn (GRPO's default), it computes a per-turn scalar `w_t ∈ [0,1]` and uses `w_t · A` for the tokens in turn `t`. `w_t` estimates the *marginal contribution* of turn `t` to the final outcome.

Two roles it fills that were previously handled by heavier machinery:

- **Denser than outcome-only RL.** Sparse reward reaches only the pivotal turns, not diluted across every token equally.
- **Cheaper than a critic or per-step PRM.** No value network, no per-step reward model, no extra rollouts. All signals come from the same distillation setup you already have.

---

## How it works

1. **Token-level evidence.** For each token `k` in turn `t`, compute the log-prob gap `Δ_{t,k} = log π_teacher(y_k|·) − log π_student(y_k|·)`. Sum within a turn to get turn-level evidence `E_t = Σ_k Δ_{t,k}`.
2. **Bayesian belief in log-odds space.** Maintain a belief `logit(p_t) = log(p_t / (1 − p_t))` that turn `t` was pivotal. Initialize `logit(p_0)` to a prior; update recursively: `logit(p_t) = logit(p_{t−1}) + E_t`.
3. **Marginal revision as credit.** The credit for turn `t` is the *change* in belief between consecutive states: `w_t = σ(logit(p_t)) − σ(logit(p_{t−1}))`, then normalize so `Σ w_t = 1`.
4. **Advantage reweighting.** Compute the trajectory advantage `A` with GRPO as usual; scale each turn's contribution by `w_t · A`. Everything else in the PPO-clipped loss stays untouched.

The recursion matters: turn `t`'s credit is *history-dependent* because `logit(p_t)` accumulates all earlier evidence. Replacing the recursion with an i.i.d. per-turn score (`w_t ∝ E_t`) or a trajectory-mean broadcast (`w_t = 1/T`) both underperform in ablations.

## Why it matters

- **Solves the pivotal-turn problem cheaply.** Long-horizon agent rollouts (ALFWorld, WebShop, τ²-Bench) have 10–30 turns; outcome-only credit is nearly noise. The Bayesian recursion catches the 2–3 pivotal turns without extra compute.
- **Doesn't compete with GRPO — composes with it.** The scheme is a reweighting on top of any policy optimizer; no changes to the loss form, no new hyperparameters beyond the prior and a normalization constant.
- **A viable substitute for training PRMs** in the specific setting where you already have a teacher-student self-distillation setup — you get most of the "per-step credit" benefit for the labeling and compute cost of nothing.

## Gotchas & tricks

- **Requires a paired teacher.** No teacher, no evidence. Common recipes: use a frozen SFT checkpoint as teacher; the on-policy student diverges from it during RL, and the gap on rollout tokens becomes the credit signal.
- **Log-odds updates can saturate.** After many turns of consistent evidence, `logit(p_t)` grows large and later marginals shrink to zero. Cap or decay `logit` between turns for very long horizons.
- **Sign of the evidence.** The log-prob gap is *not* an oracle — it says "teacher would have picked differently," which is a proxy, not a truth. When teacher and student disagree on stylistic tokens, evidence pollutes credit. Restrict evidence tokens to *decision-relevant* positions (tool-call arguments, action tokens) if you have that structure.
- **Prior matters.** A flat prior `logit(p_0) = 0` works fine when turns are of comparable length; if some turns are one-token calls and others are multi-paragraph plans, the prior should be length-adjusted or evidence should be length-normalized.

## Sources

- Paper: *AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning* — Wang et al., Tsinghua / Zhejiang U. / Meituan, 2026 — [arXiv:2608.05987](https://arxiv.org/abs/2608.05987). Introduces the log-odds recursion; evaluated on ALFWorld, WebShop, Search-QA with Qwen2.5-3B / 7B.
- Related: [prm.md](prm.md) covers the trained-per-step alternative; [../rlvr.md](../rlvr.md) is the outcome-supervision regime this densifies.
