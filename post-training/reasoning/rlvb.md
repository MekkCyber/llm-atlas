# RL with Verifiable Beliefs (RLVB)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A long-horizon RL post-training objective that shapes rewards with the environment's *exact posterior* over the hidden answer. Instead of rewarding only the final outcome, the agent is rewarded step-by-step for actions that *sharpen its belief*, and penalized for committing to an answer while the posterior is still diffuse. Targets the "lazy agent" failure mode where policies answer before gathering enough evidence.

**Prereqs:** [rlvr.md](../rlvr.md), [grpo.md](../grpo.md)
**Related:** [prm.md](prm.md) · [long-cot-rl.md](long-cot-rl.md) · [cot-reward-model.md](../cot-reward-model.md) · [_rewards.md](../_rewards.md)

---

## What it is

In long-horizon tasks, an agent takes many actions before answering. Outcome-only RL rewards the final answer; the intermediate actions get no direct signal. Empirically this produces agents that commit to an answer *before* their internal belief about the hidden ground truth has concentrated — they are "lazy" in the specific sense that they ignore the evidence they haven't yet gathered. RLVB fixes this by having the simulator expose its exact posterior over the hidden answer at every step and using the *change* in that posterior as the reward.

## How it works

Let `y` be the hidden ground-truth label, `h_t` the agent's history after step t, and `p(y | h_t)` the simulator-known posterior. Define a per-step belief entropy `H_t = H(p(· | h_t))`. The per-step shaping reward is

```
r_t = H_{t-1} - H_t      (sharpening reward)
```

so actions that reduce posterior entropy get credit proportional to the information they produce. At answer time the agent gets an outcome reward (`+1` if correct, `0` otherwise) *plus* a commit penalty proportional to the remaining entropy if the answer is wrong — committing early is only worthwhile when the posterior is already sharp. The policy is trained with GRPO-style group-relative advantages over `G` rollouts per prompt.

## Why it matters

- **Dense signal for long-horizon tasks.** Where outcome-only RL gives one scalar per trajectory, RLVB gives one per step. Variance of the policy gradient drops accordingly.
- **Named cure for a named pathology.** The "lazy agent" / early-commit failure mode is observable across reasoning agents; RLVB targets it directly.
- **Environment-grounded, not learned.** Unlike a PRM, the step-wise reward is computed from the simulator's exact posterior, not a learned verifier that can be hacked.
- **Reported gains on three VLM long-horizon benchmarks** plus a new Pro-Worlds visual puzzle suite — outperforms outcome-reward-only baselines, with the largest gap on tasks that reward early-commit shortcuts.

## Gotchas & tricks

- **Only works in posterior-observable environments.** The simulator must expose `p(y | h_t)`. Fine for synthetic worlds, Pro-Worlds-style puzzles, and Bayesian simulators; infeasible when the environment is a real deployment.
- **Entropy scale depends on the hypothesis-space size.** A toy with 2 hypotheses and one with 1000 give very different raw `H_t`. Normalize before adding to the outcome reward or the shaping term will dominate.
- **Risk of reward-hacking the posterior.** If the agent can produce actions that mechanically reveal the answer (e.g., bug in the simulator that leaks `y`), the sharpening reward is maximally gameable. Validate the simulator's information channels.
- **Commit penalty on wrong answers is load-bearing.** Without it, the agent learns to answer confidently-wrong early for the free outcome gamble; the entropy-weighted commit penalty flips that incentive.
- **Ablation vs. PRMs.** PRMs are learned, RLVB is environment-computed. In environments where both are available, RLVB is the ceiling; PRMs approximate it. Use RLVB during simulator training, then distill behavior for deployment.

## Sources

- Paper: *Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs Teaches Them to Act on What They Believe* — Feng, Wang, Poria (DeCLaRe Lab, NTU Singapore), 2026 — [arXiv:2611.32884](https://arxiv.org/abs/2611.32884).
