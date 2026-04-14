# Reasoning

*Training models to think before they answer — chain-of-thought, test-time compute, and the inference-time scaling frontier.*

---

## What This Is

Reasoning is the dominant frontier as of 2025. The core shift: instead of scaling pretraining compute, spend compute at inference time — let the model produce long chains of thought, explore multiple solutions, self-verify, and backtrack.

This area may outgrow being a subfolder; starting here because it sits naturally next to the RL post-training that produces most reasoning models.

---

## What Belongs Here

- **Chain-of-thought** — CoT prompting, self-consistency, problem decomposition.
- **Test-time compute scaling** — sampling many solutions, best-of-N, majority voting.
- **Process reward models (PRMs)** — step-level rewards vs. outcome rewards.
- **Verifier-guided search** — tree search, MCTS-style exploration, verifier models.
- **RL for reasoning** — GRPO/PPO on math & code, reward shaping for CoT.
- **Distillation** — distilling long-CoT reasoning into smaller models.
- **o1 / R1-style training** — what's publicly known, what's inferred.

## Reading Order

1. Chain-of-thought basics
2. Self-consistency & majority voting
3. Process reward models
4. RL with outcome rewards (GRPO on math)
5. Verifier-guided search
6. Distilling reasoning

---

## Related

- [../](../) — the broader post-training pipeline
- [../../inference/](../../inference/) — test-time compute is an inference problem too
- [../../case-studies/](../../case-studies/) — DeepSeek R1 and similar systems
