# RLSVR — Reinforcement Learning with Self-Verifiable Rewards
*Depth — extending RLVR to open-ended tasks via task transformation.*

**TL;DR:** RLVR only trains where an oracle grader exists (math answers, unit tests). RLSVR wraps *open-ended* tasks in a proxy environment whose internal game rules yield a hard, verifiable outcome, so the same RLVR machinery becomes applicable. The canonical instance, **SpyRL**, embeds each writing/reasoning task into a "Who Is the Spy?" self-play round: agents produce the target output under asymmetric information and vote to identify a pre-designated spy — vote correctness is the reward.

**Prereqs:** [rlvr.md](./rlvr.md), [grpo.md](./grpo.md)
**Related:** [_rewards.md](./_rewards.md), [cripo.md](./cripo.md)

---

## What it is

A training paradigm that expands the reach of verifiable-reward RL beyond intrinsically checkable domains. Rather than *finding* a verifier for an open-ended task, RLSVR **transforms the task itself** into a proxy environment where the outcome can be checked mechanically. The transformation is chosen so that succeeding at the proxy still requires producing a high-quality target output.

## How it works

1. **Wrap the base task** (summarize, write, argue) in a multi-agent game where several rollouts of the model see asymmetric information and each produce the requested output.
2. **Predetermine a hidden label** — e.g., which of the agents is the "spy." This label is not derivable from any single output but *is* derivable from the joint set of outputs given the asymmetric info.
3. **Vote.** After outputs are produced, agents (or a separate judging agent) vote for who they think the spy is.
4. **Reward = 1 if the majority identifies the true spy, else 0.** No LLM judge, no reward model — just the game's rules.
5. Feed the reward through GRPO (or any RLVR-compatible optimizer) as if it were a math-correctness signal.

The trick: because succeeding at the vote requires each agent's output to distinctively reflect its private info while still solving the task well, the proxy reward correlates with output quality — but is fully verifiable.

## Why it matters

- Eliminates the LLM-judge inference cost and bias for open-ended post-training.
- The proxy environment is a design surface: different games elicit different qualities (SpyRL emphasizes information-consistent generation; other games could target novelty, specificity, faithfulness).
- Empirically, SpyRL also improves math reasoning — the induced verifier signal is not narrowly specialized.

## Gotchas & tricks

- **Game design is the hard part.** A game whose vote is trivially solvable by the pre-designated spy (independent of output quality) collapses training.
- **Population diversity matters.** With too-homogeneous rollouts, vote signal saturates. Some noise / temperature variation across agents is essential.
- **Reward is sparse** (per game round, not per token) — combining with dense signals (see [saf-opd.md](./saf-opd.md)) is a natural extension.

## Sources

- Paper: *From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement* — Wang et al., 2026 — [arXiv:2607.23802](https://arxiv.org/abs/2607.23802)
- Code: https://github.com/wangqinsi1/SpyRL
