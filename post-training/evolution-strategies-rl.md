# Evolution Strategies for LLM Reasoning (ES-RL)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Evolution Strategies (ES) is a population-based, gradient-free post-training paradigm for LLM reasoning that had been treated as a memory-efficient-but-weaker cousin of GRPO. Recent analysis reframes it as a *distinct* paradigm with a genuine strength: **broader reasoning coverage (Pass@K)**. ES avoids the entropy collapse that hobbles GRPO's exploration, and a **sequential GRPO → ES** schedule keeps GRPO's Pass@1 gains while inheriting ES's Pass@K gains.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md), [rlvr.md](rlvr.md)
**Related:** [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [reasoning/README.md](reasoning/README.md), [../evaluation/aime.md](../evaluation/aime.md)

---

## What it is

ES perturbs the model's parameters with $N$ population samples $\theta + \sigma \epsilon_i$, evaluates each on a verifiable reasoning task, and updates $\theta$ in the direction weighted by (whitened) rewards. It never computes a policy gradient and never stores per-token log-probs — the only per-step memory cost is holding a rank-based reward vector across the population.

For LLM reasoning, the setting matches RLVR: verifiable answers (math, code) give scalar rewards, so any RL-like signal usable by GRPO is usable by ES.

## How it works

Per step:

1. **Sample population.** Draw $\epsilon_i \sim \mathcal{N}(0, I)$ for $i=1,\dots,N$. Materialize $\theta + \sigma \epsilon_i$ (typically as a delta on a shared base to avoid $N$ full copies).
2. **Evaluate.** Roll out each perturbation on the reasoning batch; verifier gives $R_i \in \{0,1\}$ or shaped.
3. **Update.** $\theta \leftarrow \theta + \frac{\eta}{N \sigma} \sum_i \tilde{R}_i \epsilon_i$, where $\tilde{R}_i$ is rank-normalized or centered.

**Why coverage improves.** The paper proves that **verifier-projected Jensen-Shannon diversity** across the population lower-bounds Pass@K. GRPO's on-policy KL-anchored update contracts to a narrow high-reward mode (entropy collapse); ES's isotropic Gaussian perturbation keeps sampling diverse solutions, so more distinct correct-answer paths survive across iterations.

**Sparse-functional-update observation.** Despite large whole-model parameter drift, held-out probes attribute ES gains to a **sparse subset of large-magnitude updates**. Broad parameter movement does not imply broad functional change; held-out tasks show no catastrophic forgetting.

**Sequential GRPO → ES.** Start with GRPO to lock in Pass@1 (exploitation), then switch to ES to raise Pass@K (exploration recovery). Dominates either alone.

## Why it matters

Reframes ES from "memory fallback for GRPO" to "complementary paradigm that fixes GRPO's coverage weakness." Practical implication: reasoning post-training pipelines should treat Pass@1 and Pass@K as separately optimizable objectives with different tools.

## Gotchas & tricks

- **Population size scales *inversely* with model size.** Larger LLMs need *smaller* $N$, not larger — contrary to the classical ES reflex. The paper attributes this to redundancy in high-dimensional parameter spaces.
- **Rank-based reward whitening** is essential; raw reward differences make the update dominated by outliers on binary verifiers.
- **Deterministic decoding at eval** hides ES's Pass@K advantage — use temperature > 0 or best-of-K sampling to see the coverage gain.
- **Antithetic pairs** ($+\epsilon_i, -\epsilon_i$) roughly halve variance at fixed compute.

## Sources

- Paper: *Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO* — Ba et al., 2026 — [arXiv:2608.27351](https://arxiv.org/abs/2608.27351)
