# Skill-Entropy RL

*Depth — an RL reward that supervises not just answers but the sequence of reasoning skills used to produce them.*

**TL;DR:** Long-horizon reasoning tasks require chaining *different* skills (algebra → planning → arithmetic → summarization). Ordinary RLVR rewards only the final answer, giving no signal on skill sequencing. Skill-Entropy RL (He et al., 2026) has the model emit both an answer and the skill label at each step, and rewards it for (a) step correctness and (b) alignment between its predicted skill sequence and a gold skill sequence. Skill Entropy is the difficulty measure: how uncertain is the next-skill distribution across a task. Qwen3-4B: **34.4% → 68.4%** on Skill²-Bench; Qwen3-1.7B: **14.6% → 40.1%**.

**Prereqs:** [../rlvr.md](../rlvr.md), [../grpo.md](../grpo.md), [long-cot-rl.md](long-cot-rl.md)
**Related:** [prm.md](prm.md) · [orm.md](orm.md) · [../rl-prompt-curation.md](../rl-prompt-curation.md)

---

## What it is

RLVR pipelines reward correctness of the final answer. That signal is sparse *and* uniform across steps — it can't tell the model that its problem was picking the wrong *kind* of reasoning step, not making an arithmetic slip. Skill-Entropy RL adds a structure reward: not just "was the answer right", but "did the skill sequence match the intended decomposition".

**Skill Entropy** is the paper's task-difficulty measure: for a task decomposition graph, entropy of the distribution over which skill to invoke next given the current state. Low entropy = obvious next skill; high entropy = the task genuinely requires switching among many skill options.

## How it works

Training-time modification to RLVR:

1. **Skill labels.** Each training task comes with a gold skill sequence — e.g. `["algebra_solve", "arithmetic", "check"]`. Skills are drawn from a fixed vocabulary (the paper uses 558 skills across 9 domains).
2. **Model output format.** At each reasoning step, the model outputs both the reasoning token stream and a skill tag (e.g. `<skill>algebra_solve</skill> ...`).
3. **Reward decomposition.** Total reward per rollout = (step-correctness reward) + (skill-entropy reward, measuring alignment of the model-predicted skill sequence with the gold sequence).
4. **Policy update.** Standard GRPO/PPO update on the composite reward. No new architecture, no new head — just an augmented reward and slightly richer output format.

The paper also derives Skill²-Bench, a benchmark scored by skill-entropy so difficulty tiers are measurable rather than intuitive.

## Why it matters

- **Adds a *structure* signal to RLVR.** RLVR's scalar answer reward is agnostic to how the model got there; skill-entropy reward names the *decomposition* as a first-class training target.
- **Reusable training signal.** The paper shows the same pipeline applied to OpenR1-Math with skill labels backfilled — skill supervision transfers to off-the-shelf reasoning datasets, not just Skill²-Bench.
- **Difficulty gets measurable.** Skill Entropy converts "this task is hard" from folklore into a computable scalar, letting benchmarks stratify by entropy rather than by author judgment.

## Gotchas & tricks

- **Skill taxonomy design is the ceiling.** 558 skills is a lot; if the taxonomy is too coarse, every task is single-skill and the reward degenerates; too fine and gold labels become noisy. Sweet spot is domain-dependent.
- **Gold skill sequences require annotation.** For synthetic training data (e.g. templated math), gold sequences are cheap; for scraped natural data they're expensive. LLM auto-labeling of gold sequences works but introduces label noise the reward inherits.
- **Model-predicted skill labels may lie.** The model can produce "correct" skill tags that don't match its actual reasoning content. Occasional consistency checks (does the tagged skill match the reasoning that follows?) are worth the compute.
- **Doesn't replace PRMs.** PRMs score every step of an unlabeled trajectory; skill-entropy needs a pre-annotated skill vocabulary. Complementary, not substitute.

## Sources

- Paper: *Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning* — He, Yang, Liu, Yang, Zhang, Wu, Yin, Wang, Arora, 2026 — [arXiv 2608.05139](https://arxiv.org/abs/2608.05139). Princeton / CMU / Toronto / UIUC / Stanford / Oxford.
