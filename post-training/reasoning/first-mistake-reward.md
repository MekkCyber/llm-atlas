# First-Mistake Reward (Cliff)

*Depth — process-style reward shaping that only asks a teacher LLM to locate the first wrong step in a rollout.*

**TL;DR:** RLVR gives one outcome reward per rollout; PRMs give per-step scores but are expensive and hackable; on-policy distillation assumes matched teacher/student patterns. **Cliff** (Han et al., 2026) observes that after the *first* mistake in a reasoning trace, subsequent steps are conditioned on an invalid prefix and give no extra learning signal. It uses an off-the-shelf teacher LLM to identify only the *first mistake* per rollout, then hands out positive token-level advantages before it and negative advantages after — no PRM training, no full-step scoring. Delivers **+15% over on-policy distillation** and **+7% over vanilla GRPO** across 12 scenarios, holding even with modestly capable teachers.

**Prereqs:** [../rlvr](../rlvr.md), [../grpo](../grpo.md), [../_rewards](../_rewards.md)
**Related:** [prm](prm.md), [orm](orm.md), [long-cot-rl](long-cot-rl.md)

---

## What it is

A reward-shaping strategy layered on top of RLVR / GRPO for reasoning post-training. Instead of one scalar per trajectory (RLVR) or one score per step from a specialized PRM (PRM), Cliff obtains a single index — *the token / step where the reasoning first goes wrong* — from an off-the-shelf LLM teacher, and converts that index into token-level advantages.

## How it works

### Step 1 — Locate the first mistake

For each on-policy rollout $\tau = (t_1, t_2, \ldots, t_T)$:

- Prompt an off-the-shelf teacher LLM with the problem and the trace.
- Ask it to identify the *first* token / step where reasoning breaks (or "no mistake" if the trace is fully correct).
- Let $k$ = first-mistake index (with $k = T{+}1$ for a fully correct trace).

The teacher does *not* need to score every step — only find the cliff. That single-decision task is much easier than PRM-style per-step judgment, which is why a modest teacher works.

### Step 2 — Shape advantages around the cliff

Given $k$ and the ground-truth outcome reward $r \in \{0, 1\}$ from a rule verifier:

$$
A_t = \begin{cases}
+ r_{\text{prefix}}, & t < k \\
- r_{\text{suffix}}, & t \ge k
\end{cases}
$$

The prefix gets positive credit (the trace was still on track), the suffix gets negative (all downstream tokens are irrelevant or harmful). Signs and magnitudes are calibrated per rollout so that a correct trajectory reduces to standard GRPO advantages.

### Step 3 — Standard GRPO update

The token-level advantages plug directly into GRPO. No PRM training, no new critic, no full-vocab teacher log-probs — the only new cost is one teacher call per rollout to locate $k$.

## Why it matters

- **Reward density with a single teacher call.** PRMs need dense labeling (humans or Monte-Carlo rollouts). Cliff replaces all of that with one classification per rollout. The cost is bounded by whatever the teacher costs — an off-the-shelf 7B open model suffices.
- **Sidesteps two known PRM failure modes.** No step-boundary ambiguity (Cliff only needs *one* boundary, not all of them), and no reward-hacking of a learned PRM (the teacher isn't in the RL loop as a differentiable reward function).
- **Complements on-policy distillation.** OPD assumes the teacher and student produce compatible traces token-by-token; Cliff only needs the teacher to *read* the student's trace and flag the cliff — no matched-pattern assumption.
- **Empirically a strong middle path.** +15% over OPD and +7% over GRPO with a modest teacher is a bigger delta than most PRM vs ORM ablations report — the mechanism is cheap enough that further scaling is plausible.

## Gotchas & tricks

- **Teacher failure mode: wrong cliff.** If the teacher misidentifies the mistake location, the shaping is worse than uniform. Sanity check: cross-validate on a held-out set of human-labeled traces; report false-cliff rate.
- **"No mistake" case.** For fully correct traces the cliff is at $T+1$ and Cliff reduces to GRPO — expected and desired, but plumbing must handle the boundary case cleanly.
- **Reward magnitude balance.** The paper carefully calibrates prefix / suffix magnitudes so early-mistake traces don't dominate the gradient. Default: normalize per-batch so mean |advantage| matches vanilla GRPO's baseline.
- **Teacher-choice tradeoff.** A smarter teacher finds the cliff more reliably; a cheaper teacher costs less per rollout. Cliff is designed for the cheap-teacher regime — the +7% vs GRPO number is with a small teacher.
- **Not for open-ended generation.** Like RLVR itself, first-mistake shaping needs a rule-verifiable final answer to anchor the "reward" $r$. Applies naturally to math, code, formal reasoning; not to helpfulness or style.
- **Adjacent to [orm](orm.md) failure diagnosis.** A generative ORM asked "at what step did this go wrong?" is essentially a Cliff teacher; the distinction is that Cliff uses the answer for advantage shaping, not for reranking.

## Sources

- Paper: *Cliff: Learning Process Rewards from the First Mistake* — Peixuan Han, Runhui Wang, Ketan Ramaneti, Jie Hao, Gerald Friedland, Chris Kong — 2026 — [arXiv:2609.02817](https://arxiv.org/abs/2609.02817).
- Related: [prm](prm.md) — for the PRM failure modes Cliff avoids.
- Related: [grpo](../grpo.md) — the RL optimizer Cliff shapes.
