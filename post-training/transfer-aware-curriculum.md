# Transfer-Aware Curriculum for Multi-Domain RLVR
*Depth — a bandit-style curriculum that picks the training domain whose gradient step benefits the *other* domains too.*

**TL;DR:** In multi-domain [RLVR](rlvr.md) (math + code + science + …) the schedule of *which domain to sample next* is usually fixed or hand-tuned. Learnability-based curricula sample the domain the policy is currently improving on — but ignore whether that step *helps or hurts* the rest. **TAC** (Transfer-Aware Curriculum) is a cheap bandit that prefers domains scoring high on **both** local advantage *and* projected-gradient alignment with the other domains, reusing signals GRPO already computes at <1 % wall-clock overhead. Beats proportional random, hand-designed, and learnability-only bandits — up to 2.8 points (10 % relative) on a 6-domain reasoning suite.

**Prereqs:** [rlvr.md](rlvr.md), [grpo.md](grpo.md), [rl-prompt-curation.md](rl-prompt-curation.md)
**Related:** [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [_rl.md](_rl.md)

---

## What it is

Multi-domain RLVR trains a single policy on a suite of verifiable-reward tasks — AIME-style math, LiveCodeBench-style code, science-QA, etc. The curriculum question: at each step, which domain should we sample the next batch from?

- **Fixed schedule / proportional random:** sample each domain at a fixed rate. Ignores learning dynamics.
- **Learnability bandit** (e.g. absolute-advantage bandit): sample the domain where the policy's current advantage is largest. Improves fast on the picked domain — but can starve or *hurt* the rest.

**Transfer-Aware Curriculum** adds a second criterion: does taking a gradient step on this domain *help* the other domains? Compute domain gradients and score each candidate by its inner product with the other domains' gradient direction. Combine with local advantage.

## How it works

For a batch of size $B$ drawn from domain $d$, GRPO already computes:
- per-domain advantage $\bar{A}_d$ (learnability signal)
- projected policy-gradient $g_d$ (from the GRPO update itself)

TAC forms a per-domain score:

$$
\text{score}(d) = \bar{A}_d \cdot \underbrace{\Bigg(1 + \alpha \cdot \frac{1}{|D|-1}\sum_{d' \neq d} \frac{\langle g_d, g_{d'} \rangle}{\|g_d\|\,\|g_{d'}\|} \Bigg)}_{\text{transfer bonus}}
$$

A softmax over $\text{score}(d)$ gives the sampling distribution for the next batch. Cosine alignment $\langle g_d, g_{d'}\rangle / (\|g_d\|\|g_{d'}\|)$ measures whether the GRPO step for $d$ points in a direction that would also improve $d'$.

Cost is negligible: the gradients are computed anyway; the extra bookkeeping is one dot product per domain pair per step. Paper reports <1 % wall-clock overhead on a 6-domain suite.

## Why it matters

- **Cross-domain transfer is asymmetric.** Math often helps code; code doesn't always help science. Learnability-only bandits miss this and over-commit to whichever domain is currently fastest to improve — even if that improvement crowds out other capabilities.
- **Best macro-averaged accuracy.** On Qwen3-1.7B and Llama3.2-3B, TAC beats proportional random, hand-designed schedules, and learnability-only bandits; up to **2.8 points** over the learnability baseline (10 % relative).
- **Robust to imbalanced mixtures.** When training data is skewed (100× more math than science), learnability-only over-commits to the dominant domain; TAC's transfer term regularizes toward diversity.
- **Ablations are clean.** Removing the transfer term causes sharp degradation — showing the term itself, not just the bandit structure, is what wins.

## Gotchas & tricks

- **The transfer bonus is unnormalized advantage-weighted.** A tiny local advantage can be inflated by a large transfer bonus; clip or normalize if you see the bandit refusing to explore a genuinely-stuck domain.
- **Projected gradients, not full backprop.** TAC uses the gradient *step being computed* — not a separate reference — so no extra backward pass per candidate domain. Cheap by construction.
- **Domain granularity matters.** Splitting math into "algebra" / "combinatorics" / "geometry" changes the transfer geometry. Coarser domains are more stable; finer are more precise but noisier.
- **Layer on top of prompt curation.** [rl-prompt-curation.md](rl-prompt-curation.md) picks *which prompts within a domain*; TAC picks *which domain*. They compose.

## Sources

- Paper: *Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR* — Yang et al., 2026 — [arXiv:2606.25178](https://arxiv.org/abs/2606.25178).
- Related: *DeepSeekMath / GRPO* — the RL algorithm whose signals TAC reuses.
- Related: *Prioritized Level Replay* — Jiang et al., 2020 — a spiritual ancestor from procgen-RL curricula.
