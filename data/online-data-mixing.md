# Online Data Mixing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Adjust the per-domain weights of a pretraining mixture *during* training, rather than fixing them up front. The Holistic Data Scheduler (HDS, 2026) frames this as continuous-control RL solved with Soft Actor-Critic, with a three-signal reward combining quality scores, inter-domain loss influence, and weight-norm dynamics. On The Pile, HDS reaches the next-best method's final validation perplexity in 44% fewer iterations and lifts MMLU 0-shot by 7.2%.

**Prereqs:** [_data-curation.md](_data-curation.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../pre-training/_lr-schedules.md](../pre-training/_lr-schedules.md) · [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

A pretraining mixture is the per-domain (web, code, math, books, …) proportion of tokens the model sees. Static mixtures pick one ratio and stick with it — DoReMi-style methods choose it offline by training many small proxies. Online data mixing instead treats the mixture proportions as a controller variable that updates throughout training, responding to the model's current state.

## How it works

HDS frames mixture control as a continuous-control RL problem:

- **State**: per-domain loss, gradient statistics, and weight-norm features at the current step.
- **Action**: a vector of mixture proportions (continuous, sums to 1 after softmax).
- **Reward**: weighted sum of three signals capturing complementary perspectives:
  1. **Data-driven** — domain-level quality scores (encourages high-quality domains).
  2. **Loss-driven** — inter-domain influence (encourages domains whose inclusion lowers loss elsewhere).
  3. **Model-driven** — weight-norm dynamics (encourages mixtures that keep weights well-conditioned).

Soft Actor-Critic (SAC) is used as the policy optimizer. Its stochastic policy and entropy regularization are well-suited to the high-dimensional mixture space where exploration matters and exploitation is cheap.

## Why it matters

- Static mixtures bake in offline guesses; the model's needs change as it learns (math becomes more useful after format is solidified, etc.). Online schedulers can track those needs.
- The three-signal reward is a *template*: any new signal (gradient noise, domain-level perplexity, downstream proxy eval) can be plugged in without redesigning the controller.
- 44% iteration savings to reach the same perplexity is a large lever for compute-bound pretraining runs; the +7.2 MMLU 0-shot gain at fixed final perplexity is real signal that the mixture trajectory matters, not just the endpoint.

## Gotchas & tricks

- SAC has its own hyperparameters (temperature, target networks). Bad SAC tuning shows up as oscillating mixtures, which destabilizes the training loss.
- The reward weights for the three signals are tunable and shift the resulting model's bias — heavier weight-norm reward gives smoother training; heavier quality reward gives sharper capability but riskier loss curves.
- Works in the regime where domain assignments are clean. Web crawls with mixed-domain documents need a pre-step to attribute tokens.
- Open question: how the learned schedule transfers across model scales — HDS was validated on small to mid-scale LLMs.

## Sources

- Paper: *Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning* — 2026 — [arXiv:2606.24133](https://arxiv.org/abs/2606.24133).
- Reference for comparison: *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining* — Xie et al., 2023.
- Reference for comparison: *Soft Actor-Critic* — Haarnoja et al., 2018.
