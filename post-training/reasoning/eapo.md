# EAPO — Entropic Advantage Policy Optimization
*Depth — token-level credit assignment for RLVR that reinforces surprising success and penalizes repeated failure using entropy alone.*

**TL;DR:** RLVR gives an outcome-only reward, so per-token credit is coarse. EAPO redistributes the response advantage across tokens using policy entropy, but asymmetrically: reinforce **high-entropy tokens in correct responses** (surprising success) and penalize **low-entropy tokens in wrong responses** (repeated failure), while attenuating penalties at uncertain positions inside failed responses so recovery paths are not crushed.

**Prereqs:** [../rlvr.md](../rlvr.md), [../grpo.md](../grpo.md)
**Related:** [long-cot-rl](long-cot-rl.md), [prm](prm.md), [orm](orm.md)

---

## What it is

A token-level modification of the RLVR / GRPO advantage that requires no auxiliary model, no extra rollouts, and no privileged information. It takes the response-level advantage the algorithm already computes and reweights it token-by-token using the normalized policy entropy at each position, with a sign-aware asymmetry.

## How it works

For each rollout with response-level advantage `A` (positive for success, negative for failure) and per-token policy entropy `H_t` (normalized to `[0, 1]`):

- **Success (`A > 0`):** scale the reward on token `t` by `H_t`. Tokens where the model was *uncertain* but the response worked out get the strongest reinforcement — the assumption is that success under uncertainty is fragile and needs positive credit.
- **Failure (`A < 0`):** scale the penalty by `(1 - H_t)`. Confident tokens in failed rollouts get the strongest penalty ("you were sure and wrong"); uncertain tokens are attenuated, preserving alternative continuations for future exploration.
- **Symmetric mean.** Normalize so the token-level advantages sum to `A` — this keeps the response-level update magnitude identical to plain GRPO/RLVR.

The credit derives entirely from rollout signals already available; no separate value head or PRM is trained.

## Why it matters

Fine-grained credit assignment usually needs extra machinery — a PRM, extra sampled rollouts, or self-consistency traces. EAPO gets a useful credit distribution *for free* from the rollout's own entropy. Practically, it also promotes exploration: because low-entropy failure is punished hardest, the model is pushed away from confidently-wrong modes; because high-entropy success is reinforced hardest, it is pushed to *keep* the exploratory branches that worked. Reported results: best overall performance across the tested reasoning tasks on both base and reasoning backbones, with broader problem coverage and more diverse candidate answers than plain GRPO/RLVR.

## Gotchas & tricks

- **Entropy normalization matters.** Use the maximum-possible entropy over the vocabulary (or a token-normalized proxy) rather than raw entropy — otherwise vocabulary size interacts with the asymmetry.
- **Interaction with temperature.** Cold-sampled rollouts have low entropy everywhere, so EAPO collapses to plain GRPO. Warm sampling during rollouts is required for the redistribution to be non-trivial.
- **Compatible with CIS.** [calibrated-importance-sampling](../calibrated-importance-sampling.md) corrects the *update* variance while EAPO reshapes the *advantage* — stack them.
- **Length-agnostic.** Because normalization keeps the response-level total unchanged, EAPO doesn't induce a length bias by itself. But if you combine it with a length penalty ([length-penalty](length-penalty.md)), verify the two don't work against each other.

## Sources

- Paper: Surprising Success, Repeated Failure: Entropy-Guided Credit Assignment for Exploration in LLM Reasoning — Yeo, Kang, Lee, Park, Baek, Hwang, 2026 — [arXiv:2609.33781](https://arxiv.org/abs/2609.33781)
