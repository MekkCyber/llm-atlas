# MMLU-Pro
*Depth — the harder, less-saturated successor to MMLU.*

**TL;DR:** ~**12,000 questions** across 14 broad domains, **10 answer choices** per question (vs MMLU's 4), more reasoning-heavy questions, less saturated at the frontier. Drops scores by **16–33 points** vs MMLU for the same model — restoring discrimination at the top. Introduced by Wang et al. (NeurIPS 2024, arXiv 2406.01574) as the de-facto successor to the now-saturated MMLU. Llama 3.1 405B: **73.3%**; Claude 3.5 Sonnet: 77.0%; GPT-4o: 74.0%.

**Prereqs:** [mmlu](mmlu.md)
**Related:** [gpqa](gpqa.md) · [aime](aime.md)

---

## What it is

Wang et al., *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark*, NeurIPS 2024, arXiv 2406.01574.

Designed specifically to address MMLU's two main failure modes at the frontier:
1. **Saturation** — top models cluster at 86–90%, within labeling-error noise.
2. **Shallow questions** — many MMLU questions can be answered by shallow pattern-matching rather than reasoning.

MMLU-Pro's fixes:
- **10 answer choices instead of 4** — random baseline drops from 25% → 10%, halving guessability.
- **More reasoning-heavy questions** — filtered toward those requiring multi-step inference.
- **Human re-curation** — flagged and fixed mislabeled questions from MMLU's source.
- **14 domains** (vs MMLU's 57 subjects) — reorganized into broader clusters: Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology, Other.

---

## How it works as an LLM eval

### Format

- **~12,000 test questions** (paper: 12,032).
- 10 answer choices A–J per question.
- Output: single letter.
- Grading: exact match.

### Scoring conventions

- **5-shot CoT**: the paper's default. Use 5 demonstration examples with chain-of-thought reasoning shown.
- **0-shot CoT**: for instruction-tuned / reasoning models.
- **No CoT**: some evaluations; typically scores worse because MMLU-Pro questions benefit from reasoning.

### Typical harness

- Open LLM Leaderboard v2 (HF).
- `lm-eval-harness` with `mmlu_pro` task.
- Original repo: https://github.com/TIGER-AI-Lab/MMLU-Pro.

---

## Why it matters

- **Restores discrimination at the top.** MMLU's top-model cluster breaks into a spread on MMLU-Pro.
- **Forces reasoning.** 10 options + harder questions mean shallow pattern-matching no longer works.
- **Replaces MMLU as the default.** By late 2024, MMLU-Pro has become the headline "general capability" number in new tech reports.
- **Still a multiple-choice benchmark.** Compared to AIME / MATH-500 (open-ended) or HumanEval (code execution), MMLU-Pro retains multiple-choice's simplicity. Grading is trivial.

---

## Gotchas & tricks

- **CoT matters.** MMLU-Pro is designed to reward CoT reasoning. Models that don't CoT score noticeably lower.
- **Still saturatable.** As of 2025, top reasoning models (o1, Claude 3.5 Sonnet Extended) hit low-80s. The benchmark has maybe 1–2 years of headroom before it saturates too.
- **Domain imbalance.** Math, physics, and engineering questions are the hardest discriminators; law and economics questions saturate earlier. Per-domain breakdowns matter.
- **Answer-letter bias.** 10 options are more prone to positional bias than 4. Some harnesses shuffle answer order to mitigate.
- **Contamination risk.** Test set is public; scores on frontier models are likely inflated by memorization. Cannot be fixed post-release.
- **Compare against same harness.** lm-eval-harness vs simple-evals vs the official TIGER harness report slightly different numbers for the same model. Check.

---

## Typical modern numbers (5-shot CoT)

| Model | MMLU-Pro |
|---|---|
| Claude 3.5 Sonnet | 77.0% |
| GPT-4o | 74.0% |
| Llama 3.1 405B | 73.3% |
| Llama 3.1 70B | 66.4% |
| o1 (full) | >80% |
| DeepSeek V3 | ~65% |
| DeepSeek R1 | 84.0% |
| Llama 3.1 8B | 48.3% |
| Random baseline | 10% |

---

## Sources

- Paper: *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark* — Wang et al., NeurIPS 2024, arXiv 2406.01574.
- Repo: https://github.com/TIGER-AI-Lab/MMLU-Pro.
- Original [mmlu](mmlu.md) for context.
