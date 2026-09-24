# Bellman Policy Optimization (BPO)
*Depth — critic-free RLVR derived from Policy Mirror Descent via the Bellman equations.*

**TL;DR:** BPO is a critic-free RL algorithm for language-model RLVR that reformulates Policy Mirror Descent (PMD) as a **trajectory-level** objective. For autoregressive generation with terminal rewards, applying the Bellman equations to PMD's per-state objective removes the need to estimate intermediate state values while provably preserving the same unique optimum. The practical loss is a policy-gradient surrogate whose per-token weight is a *mismatch-correction* term — a smoothed ratio of complementary token probabilities under the current and reference policies.

**Prereqs:** [ppo.md](../ppo.md), [grpo.md](../grpo.md), [online-policy-mirror-descent.md](./online-policy-mirror-descent.md)
**Related:** [rlvr.md](../rlvr.md), [_rl.md](../_rl.md), [long-cot-rl.md](./long-cot-rl.md)

---

## What it is

An RLVR algorithm sitting in the same PMD family as Kimi k1.5's online policy mirror descent, but derived by pushing the mirror-descent objective through the Bellman equations for terminal-reward MDPs (rewards only at the final answer, none in the middle of a chain-of-thought). The reformulation collapses the per-state PMD objective into a single **trajectory-level** term. No critic. No per-state value estimation. Same unique global optimum as the original PMD objective (proved in the paper).

Sits alongside [GRPO](../grpo.md) and [online policy mirror descent](./online-policy-mirror-descent.md) as a third value-network-free RLVR algorithm — GRPO is derived from PPO, online-PMD from mirror descent with an $\ell_2$ regression surrogate, BPO from the same PMD but via Bellman.

## How it works

The starting point is standard PMD: at iteration $i$, freeze $\pi_{\theta_i}$ as the reference and solve
$$\max_\theta\; \mathbb{E}_{s,a\sim \pi_\theta}[Q^{\pi_\theta}(s,a)] - \tau \cdot \mathrm{KL}(\pi_\theta\,\|\,\pi_{\theta_i}).$$

For autoregressive generation, states $s$ are prefixes and the *only* reward is at the terminal step. Under this structure, the Bellman equations let you rewrite $\mathbb{E}_{s,a}[Q^{\pi_\theta}]$ as a trajectory-level expectation of the terminal reward, eliminating intermediate $V(s_t)$ terms. The paper proves the reformulated objective and the original PMD have the same unique optimizer.

Approximating the resulting objective yields the BPO loss: a trajectory-level policy-gradient surrogate whose *mismatch-correction weight* per token is a smoothed ratio of complementary token probabilities under $\pi_\theta$ vs $\pi_{\theta_i}$. Rollouts come from $\pi_{\theta_i}$; correcting for the off-policy gap is what that smoothed ratio does — a Bellman-derived cousin of PPO's importance-ratio clip.

## Why it matters

- **Same critic-free family as GRPO / online-PMD, but with a Bellman derivation.** If you buy the PMD-mirror-descent framing, BPO is arguably the "cleanest" trajectory-level instantiation for terminal-reward RLVR.
- **Skips state-value estimation entirely.** Even relative to GRPO, there's no group baseline heuristic — the mismatch-correction weight replaces it.
- **Provable equivalence** to per-state PMD is a rare thing to have in this niche; algorithms in this family are usually justified empirically.

## Gotchas & tricks

- Requires **terminal-reward** MDPs — dense per-token rewards break the Bellman reformulation. Standard RLVR is fine; PRM-style dense signals are not.
- The mismatch-correction ratio can spike numerically when a token has near-zero probability under $\pi_{\theta_i}$. The paper's "smoothed" form is load-bearing; open implementations should port the smoothing rather than raw ratios.
- Reference-update schedule (per-iteration vs rolling) mirrors the PMD-vs-GRPO split; the paper's headline results use per-iteration references.

## Sources

- Paper: *Bellman Policy Optimization* — Song, Xu, Zhang, Bing (Alibaba DAMO), 2026 — [arXiv:2609.15987](https://arxiv.org/abs/2609.15987).
- Related: *Kimi k1.5* — Moonshot AI, 2025 — the online-PMD sibling.
- Related: *DeepSeekMath* — Shao et al., 2024 — GRPO, the PPO-derived cousin.
