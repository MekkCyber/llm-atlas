# Answer-Backtracked Credit Assignment (ABC)

*Depth — a way to turn sparse outcome rewards on long-horizon agent trajectories into dense step-level supervision without training a process reward model.*

**TL;DR:** Long-horizon search agents get one reward at the end of a trajectory (right answer / wrong answer), so uniform SFT and RL treat every intermediate action — search, retrieve, verify, integrate — identically. ABC re-scores each step *after* the rollout by backtracking from the final answer: which steps produced evidence that ended up in the answer, and which were dead ends or redundant. Those per-step scores become the SFT weight or RL advantage, giving dense supervision even on failed trajectories. Introduced by Lu et al. (SJTU) 2026.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md) · [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md) · [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

A retrospective credit-assignment scheme for search-agent training. A rollout is a sequence of `(action, observation)` steps ending with a final answer that either matches ground truth or not. Outcome-only training assigns the same reward to every step, wasting most of the signal from failures. ABC replaces the single outcome scalar with a *per-step* score computed by looking backward from the answer.

## How it works

1. **Rollout** the agent on a task with checkable ground truth. Store the full trajectory `T = [(a_1, o_1), ..., (a_n, o_n), answer]`.
2. **Backtrack** from `answer` step by step. For each step `i`, ask: did the evidence produced by step `i` appear in the answer? Was step `i`'s observation cited, verified, or otherwise used downstream? Score `s_i ∈ [-1, 1]` accordingly — positive for useful, near-zero for redundant, negative for misleading.
3. **Convert to training signal.** For SFT, use `s_i` as a per-step loss weight. For RL (GRPO-style), use `s_i` as the per-step advantage instead of broadcasting one trajectory reward across all tokens.
4. **Rescue failed trajectories.** Steps that were locally correct (e.g. a good sub-query) in a failed trajectory still get positive `s_i` — the paper's key claim is that this dense signal is what standard RL throws away.

The re-scoring is done by an evaluator (LLM-based or rule-based) that has both the trajectory and the ground-truth answer — a cheap "look-ahead" not available at rollout time.

## Why it matters

- **Verifier-free process rewards.** PRMs need labeled process data; ABC bootstraps from outcome supervision plus a retrospective scorer.
- **Signal from failures.** Long-horizon agents fail often; ABC keeps their partial successes as gradient rather than discarding them.
- **Sample-efficiency for expensive rollouts.** Multi-step search rollouts (with real HTTP calls) are the bottleneck of agent RL — extracting more gradient per rollout is a first-order win.

## Gotchas & tricks

- **Evaluator matters.** The retrospective scorer *is* the reward model; a bad scorer teaches the agent to game its backtracking heuristic. Use a stronger model than the agent whenever feasible.
- **Positive credit inside failed trajectories can encourage a "helpful but wrong" mode** — the agent learns to do many locally-plausible steps that don't compose into a correct answer. Combine with an outcome-only bonus to keep the answer-correctness incentive alive.
- **Non-causal.** ABC uses the future to score the past. It's fine at training time but does not give the *deployed* agent any new capability by itself.
- **Distinct from PRM.** A PRM predicts step-level goodness *without* knowing the answer; ABC uses the answer. In principle ABC-scored trajectories can be used to distill a PRM.

## Sources

- Paper: *Training Long-Horizon Search Agents via Answer-Backtracked Credit Assignment* — Lu, Ye, Wang, Du, Jin, Liu, Chen, 2026 — [arXiv 2608.05102](https://arxiv.org/abs/2608.05102).
