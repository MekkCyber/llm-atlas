# ComPO — Zeroth-Order Preference Optimization
*Depth — preference alignment via a comparison oracle and zeroth-order updates, sidestepping DPO's likelihood-displacement pathology.*

**TL;DR:** DPO and its variants fit a differentiable loss on preference pairs. On pairs with a **small likelihood margin** the loss can push the preferred completion's likelihood *down* — a documented pathology called likelihood displacement. **ComPO** (Comparison-based Preference Optimization) uses only a **comparison oracle** (which of two is preferred) and extracts *directional* update information via **zeroth-order optimization** — finite-difference gradients that don't require a differentiable preference loss on the pair. Offline and online variants; provable convergence under smoothness + gradient-sparsity + oracle-compatibility conditions.

**Prereqs:** [dpo](dpo.md), [ppo](ppo.md)
**Related:** [_post-training](_post-training.md) · [_rewards](_rewards.md) · [grpo](grpo.md)

---

## What it is

A rethink of preference optimization: treat the labeler as an oracle that returns a *comparison bit*, not a scalar. If the loss-based approach misbehaves on small-margin pairs (because the loss surface doesn't reliably point at the winning side), fall back to a method whose only requirement is that the oracle be *directionally correct*.

Zeroth-order optimization estimates a gradient without derivatives:
$$
\widehat{\nabla f(\theta)} \approx \frac{f(\theta + \epsilon u) - f(\theta - \epsilon u)}{2\epsilon} \cdot u
$$
for a random direction $u$. Replace $f(\theta + \epsilon u) - f(\theta - \epsilon u)$ with the comparison-oracle's answer to *"which of $\pi_{\theta + \epsilon u}$ and $\pi_{\theta - \epsilon u}$ is preferred?"* and you get a preference-oracle gradient estimator.

## How it works

- **Offline ComPO.** For each preference pair $(x, y_w, y_l)$, sample a perturbation $u$ of the policy parameters. Query the oracle to compare policies $\pi_{\theta + \epsilon u}$ and $\pi_{\theta - \epsilon u}$ (in practice, use $y_w$ vs $y_l$ as a stand-in for the oracle). Update $\theta$ in the direction the oracle prefers.
  - Convergence guarantee under smoothness of the latent objective, gradient sparsity, and compatibility between the comparison oracle and the latent gradient.
- **Online ComPO.** Retain the comparison-oracle mechanism; add a reverse-KL regularizer to a reference policy, estimated over unlabeled policy generations. Performance guarantee established under local coverage and in-distribution pairwise-reward accuracy.

## Why it matters

- **Addresses DPO's most-documented failure mode** without changing the loss shape (SimPO, DPO-Positive) or the data (IPO). Roots the fix in the *oracle model*, not the loss.
- **Loss-agnostic.** ComPO doesn't assume Bradley-Terry, log-ratio parameterization, or any pair-level scalar reward. Any comparison signal works.
- **Wins across families.** Reported improvements on Mistral, Llama, Gemma-2, Qwen3, Gemma-3, including length-controlled win rates. Pair-level diagnostics show mitigation of likelihood displacement.

## Gotchas & tricks

- **Variance of ZO estimates scales with dim($\theta$).** Naive ZO is impractical at LLM scale. ComPO relies on gradient-sparsity — only a small subspace of directions matters — to keep sample complexity feasible.
- **Perturbation size $\epsilon$ matters.** Too small → oracle answer is noise. Too large → policies differ enough that the oracle judges a *different* preference than at $\theta$.
- **Online coverage matters.** The online scheme's guarantee assumes the reference policy's rollouts cover the same behavior distribution as preferences. Off-distribution reference collapses the guarantee.
- **Not a replacement for verifier-RL.** RLVR / GRPO have *scalar* rewards, not comparison oracles. ComPO addresses the comparison-only regime.

## Sources

- Paper: *A Zeroth-Order Paradigm for LLM Preference Alignment* — Chen, Chen, Yin, Lin, Columbia / Alibaba DAMO, 2026, arXiv 2609.19144.
