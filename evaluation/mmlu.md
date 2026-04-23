# MMLU — Massive Multitask Language Understanding
*Depth — the 57-subject multiple-choice knowledge eval that was the standard capability benchmark of the GPT-3 / GPT-4 era.*

**TL;DR:** **57 subjects × ~250 questions each = 15,908 total**, 4-way multiple choice. Covers elementary through professional-level material across humanities, social sciences, STEM, and "other". Default 5-shot exact-match on the letter (A/B/C/D). Introduced by Hendrycks et al. 2020 as a broad knowledge benchmark; was the headline "GPT-class" capability number through 2023. **Now saturated** at the frontier (86–92% across GPT-4o, Claude 3.5 Sonnet, o1, Gemini Ultra — within <2 pts of each other, and of the ~89.8% human expert estimate). Successor is **MMLU-Pro** (10 answer choices, harder, less saturated).

**Prereqs:** *(none)*
**Related:** [math500](math500.md) · [aime](aime.md)

---

## What it is

Hendrycks et al., *Measuring Massive Multitask Language Understanding*, ICLR 2021, arXiv 2009.03300.

- **57 subjects**, grouped into four high-level categories:
  - **Humanities** — philosophy, history, law, moral disputes, etc.
  - **Social sciences** — economics, psychology, US foreign policy, etc.
  - **STEM** — physics, chemistry, biology, abstract algebra, machine learning, etc.
  - **Other (business, health, misc.)** — accounting, medicine, business ethics, etc.
- **Range**: from elementary ("elementary mathematics") to professional ("professional medicine", "professional law").
- **Sizes** (paper Section 3):
  - **5 questions per subject `dev`** (5 × 57 = 285) — used as few-shot exemplars.
  - **1,540 `validation`** — for hyperparameter tuning.
  - **14,079 `test`** — the eval set.
  - **Minimum 100 test examples per subject.**
- **Format**: each question has 4 answer choices labeled A/B/C/D. Exactly one is correct.
- **Metric**: accuracy. Macro-average across subjects (weight-equal per subject) is the common aggregate; some evaluators use micro-average (weight-equal per question).

---

## How it works as an LLM eval

### Scoring conventions

- **5-shot** (paper default): the 5 dev examples for the subject are shown as few-shot demonstrations before each test question.
- **0-shot**: modern instruction-tuned models are often evaluated 0-shot — just the question.
- **0-shot with CoT**: reasoning models (o1, R1, Kimi k1.5) evaluate 0-shot with chain-of-thought prompting; typically scored via letter extraction from the CoT.
- **Log-probability** vs **generation**: two implementations exist. Log-prob computes `log P(A|prompt)` for each choice and picks the argmax. Generation samples the model and extracts the chosen letter. Both appear in the literature and can give different numbers; generation is now standard for instruction-tuned / reasoning models.

### Output

One letter: A, B, C, or D. Grading is exact match on the letter.

### Typical harness

- EleutherAI's `lm-eval-harness` — for log-prob style.
- OpenAI's `simple-evals` — for generation style.
- HELM, Hugging Face Open LLM Leaderboard — each has its own slightly different conventions.

Numbers across harnesses can vary 1–3 pp for the same model. Always check which harness the paper used.

---

## Why it matters

- **Historical anchor.** MMLU was the headline capability number for every frontier LLM release from GPT-3 (43.9%) through GPT-4 (86.4%) through the early Claude / Gemini / open-LLM era. Every tech report 2020–2024 reports it.
- **Breadth.** No other pre-2024 benchmark covered 57 distinct subjects. MMLU was the first eval that could say "this model is broadly knowledgeable" in a defensible way.
- **Cheap and reproducible.** Multiple choice, exact match, no LLM judge, standard splits. A 7B model evaluates in minutes; a 70B in hours.

---

## Gotchas & tricks

- **Saturated.** Frontier models cluster 86–92%. Differences in this range are within harness noise. MMLU no longer discriminates top models; use MMLU-Pro, GPQA, or reasoning-specific benchmarks instead.
- **Contamination.** The test set has been public since Sept 2020. Every pretraining corpus has seen it. MMLU-Redux (Gema et al. 2024) found ~6–14% of questions had labeling errors, which caps the achievable ceiling around 86–88% even ignoring contamination. Scores above that are model-specific noise or over-fit.
- **Multiple-choice guessability.** 4 options = 25% random baseline. Models with letter biases (preferring "A" or the longest option) can score above random without genuine knowledge. Modern harnesses sometimes shuffle answer order to detect this.
- **Log-prob vs generation gives different numbers.** Log-prob is what GPT-3 and Hendrycks used. Instruction-tuned models don't always emit the correct token as the single argmax (chat formats interfere). Generation with letter extraction is the fair modern comparison.
- **CoT helps some subjects, hurts others.** For STEM and math, 0-shot CoT typically improves scores. For professional law and philosophy, CoT sometimes overthinks and decreases accuracy. Reasoning-model numbers usually report the better of with/without CoT.
- **Macro vs micro average.** The 57 subjects have different sizes (min 100 test questions, some have more). Macro-average weights each subject equally; micro weights each question equally. Papers usually don't specify. Differences are typically <1 pp but can be noisy on imbalanced models.
- **Per-subject results matter.** A 90% macro-average can hide chance-level performance on specialized subjects. Good reports include per-subject breakdown.

---

## Successors and companions

- **MMLU-Pro** (Wang et al., NeurIPS 2024, arXiv 2406.01574): ~12,000 questions, **10 answer choices instead of 4**, more reasoning-heavy, 14 domains. Scores drop 16–33 pp vs MMLU. Current de-facto successor.
- **MMLU-Redux** (Gema et al. 2024): re-labeled subset of MMLU with corrected labels. Used to estimate ceiling.
- **GPQA / GPQA-Diamond** (Rein et al. 2023) — graduate-level science, deliberately contamination-hard.
- **MMMU** (Yue et al. 2024) — multimodal extension, image + text.
- **C-Eval** — Chinese-language analog.
- **BBH (BIG-Bench Hard)** — harder subset of BIG-Bench with reasoning focus.

For modern capability evaluation, MMLU is reported as a sanity check / historical reference, and the real comparison is on MMLU-Pro + GPQA + domain-specific reasoning benchmarks.

---

## Typical modern numbers

| Model | MMLU (5-shot) |
|---|---|
| o1 | ~92% |
| GPT-4o | 87.2% |
| Claude 3.5 Sonnet | 88.3% |
| Gemini 1.5 Pro | ~85% |
| Kimi k1.5-short | 87.4% |
| DeepSeek-V3 | 88.5% |
| DeepSeek-R1 | 90.8% |
| Llama-3.1-405B-Instruct | 88.6% |
| Qwen2.5-72B-Instruct | 85.3% |
| Llama-3.1-8B-Instruct | ~68–73% |
| Random baseline | 25% |
| Human expert estimate (paper) | ~89.8% |

---

## Sources

- Paper: *Measuring Massive Multitask Language Understanding* — Hendrycks, Burns, Basart, Zou, Mazeika, Song, Steinhardt, ICLR 2021, arXiv 2009.03300 — the original MMLU paper.
- Repo: https://github.com/hendrycks/test — official code and data.
- Paper: *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark* — Wang et al., NeurIPS 2024, arXiv 2406.01574 — MMLU-Pro.
- Paper: *Are We Done With MMLU?* — Gema et al., 2024 — MMLU-Redux analysis of labeling errors.
- Paper: *GPQA: A Graduate-Level Google-Proof Q&A Benchmark* — Rein et al., 2023, arXiv 2311.12022.
- OpenAI `simple-evals` repo — https://github.com/openai/simple-evals — reference generation-style harness.
