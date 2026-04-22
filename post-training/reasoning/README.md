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

### Learned reward models for reasoning

- **[orm](orm.md)** — outcome reward model. Trained on final-answer correctness labels. Used for best-of-N reranking and RL reward. Cobbe 2021 is the origin.
- **[prm](prm.md)** — process reward model. Trained on per-step labels. Used for reranking, search, and (less reliably) dense RL reward. Uesato 2022 and Lightman 2023 are the canonical references.

### Search-augmented reasoning

- **[mcts](mcts.md)** — Monte Carlo Tree Search applied to LLM reasoning. Classical UCT/AlphaZero machinery adapted to LLMs with mixed success. DeepSeek-R1 tried and abandoned it; other groups continue to get gains with step-level actions + learned PRMs.

---

## Reading Order

1. [long-cot-rl](long-cot-rl.md) — the R1-Zero thesis; reasoning emerges from RL with outcome-only rewards on a strong base.
2. [orm](orm.md) — the simplest learned reward model for reasoning.
3. [prm](prm.md) — what PRMs are, why they sometimes beat ORMs for reranking, and why they struggle as RL rewards.
4. [mcts](mcts.md) — tree-search-based reasoning, and why the R1 team rejected it.

---

## Related

- [../](../) — general post-training: [GRPO](../grpo.md), [RLVR](../rlvr.md), [rejection-sampling](../rejection-sampling.md). Reasoning recipes are applications of these general techniques.
- [../_rl](../_rl.md) — RL fundamentals that every depth file here assumes.
- [../_rewards](../_rewards.md) — the full landscape of reward signals, including non-reasoning options.
- [../../inference/](../../inference/) — test-time compute is also an inference problem.
- [../../case-studies/](../../case-studies/) — [DeepSeek-R1](../../case-studies/deepseek-r1.md) is the canonical end-to-end reasoning-model case study.
