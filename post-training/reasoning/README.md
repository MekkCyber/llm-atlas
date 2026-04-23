# Reasoning

*Training models to think before they answer — long-CoT RL, process and outcome rewards, search-augmented inference, and the frontier of reasoning capability.*

---

## What This Is

Reasoning is the dominant frontier as of 2025. The core shift: instead of scaling pretraining compute, spend compute at inference time — let the model produce long chains of thought, explore multiple solutions, self-verify, and backtrack. Most modern reasoning recipes combine **RL with verifiable rewards** (training signal) with **long CoT** (inference pattern) — the model learns to think longer, reflect, and self-check via reward pressure alone.

This folder sits inside [post-training/](..) because most of the techniques *are* post-training techniques applied with reasoning-specific signals and rewards.

---

## Depth Files

### Elicitation via RL

- **[long-cot-rl](long-cot-rl.md)** — pure RL from a strong base grows long, reflective reasoning traces without any SFT demonstrations. R1-Zero's thesis. The generalizable recipe.

### RL objectives for long-CoT

- **[online-policy-mirror-descent](online-policy-mirror-descent.md)** — Kimi k1.5's RL objective. KL-regularized expected reward solved via an ℓ₂-regression surrogate; the principled sibling to [grpo](../grpo.md).

### Reward shaping for long-CoT

- **[length-penalty](length-penalty.md)** — the group-relative, asymmetric `±0.5` length reward that fights overthinking in long-CoT RL. From Kimi k1.5.

### Distilling long-CoT into short-CoT

- **[long2short](long2short.md)** — four methods (model merge, shortest rejection sampling, DPO, long2short RL) for compressing long-CoT capability into a token-efficient short-CoT model. From Kimi k1.5.

### Learned reward models for reasoning

- **[orm](orm.md)** — outcome reward model. Trained on final-answer correctness labels. Used for best-of-N reranking and RL reward. Cobbe 2021 is the origin.
- **[prm](prm.md)** — process reward model. Trained on per-step labels. Used for reranking, search, and (less reliably) dense RL reward. Uesato 2022 and Lightman 2023 are the canonical references.

### Search-augmented reasoning

- **[mcts](mcts.md)** — Monte Carlo Tree Search applied to LLM reasoning. Classical UCT/AlphaZero machinery adapted to LLMs with mixed success. DeepSeek-R1 tried and abandoned it; other groups continue to get gains with step-level actions + learned PRMs.

---

## Reading Order

1. [long-cot-rl](long-cot-rl.md) — the R1-Zero thesis; reasoning emerges from RL with outcome-only rewards on a strong base.
2. [online-policy-mirror-descent](online-policy-mirror-descent.md) — Kimi k1.5's mirror-descent derivation of the RL objective; pair with [grpo](../grpo.md) to see two algorithm choices for the same problem.
3. [length-penalty](length-penalty.md) — the reward-shaping trick that stops long-CoT RL from overthinking.
4. [long2short](long2short.md) — four methods for compressing long-CoT into short-CoT.
5. [orm](orm.md) — the simplest learned reward model for reasoning.
6. [prm](prm.md) — what PRMs are, why they sometimes beat ORMs for reranking, and why they struggle as RL rewards.
7. [mcts](mcts.md) — tree-search-based reasoning, and why the R1 team rejected it.

---

## Related

- [../](../) — general post-training: [GRPO](../grpo.md), [RLVR](../rlvr.md), [rejection-sampling](../rejection-sampling.md). Reasoning recipes are applications of these general techniques.
- [../_rl](../_rl.md) — RL fundamentals that every depth file here assumes.
- [../_rewards](../_rewards.md) — the full landscape of reward signals, including non-reasoning options.
- [../../inference/](../../inference/) — test-time compute is also an inference problem.
- [../../case-studies/](../../case-studies/) — [DeepSeek-R1](../../case-studies/deepseek-r1.md) and [Kimi k1.5](../../case-studies/kimi-k1-5.md) are the two canonical end-to-end long-CoT-reasoning case studies.
