# MGSM — Multilingual Grade School Math
*Depth — GSM8K's math word problems translated into 10 non-English languages.*

**TL;DR:** **250 problems per language**, translated from GSM8K (Cobbe 2021), across **10 non-English languages** (Bengali, Chinese, French, German, Japanese, Russian, Spanish, Swahili, Telugu, Thai). Evaluates whether math reasoning transfers from English to other languages. Introduced by Shi et al. (Google, ICLR 2023, arXiv 2210.03057). Still actively reported; Llama 3.1 405B scores **91.6%** (averaged across languages). Approaching saturation for frontier English-centric models on high-resource languages.

**Prereqs:** *(none)*
**Related:** [math500](math500.md) · [aime](aime.md) · [mmlu](mmlu.md)

---

## What it is

Shi et al., *Language Models are Multilingual Chain-of-Thought Reasoners*, ICLR 2023, arXiv 2210.03057.

- **GSM8K source**: Cobbe et al. 2021's 8,500 grade-school math word problems. Integer answers.
- **MGSM**: 250 problems from GSM8K test set, **translated by professional translators** into 10 languages:
  - Bengali (bn), Chinese (zh), French (fr), German (de), Japanese (ja), Russian (ru), Spanish (es), Swahili (sw), Telugu (te), Thai (th).
- English source questions also included as baseline, so effectively 11 languages × 250 = 2,750 problems.
- Answers remain as Arabic numerals (not written-out words).

### Purpose

Tests whether math reasoning — a capability developed largely from English-heavy training data — **transfers** to other languages, or degrades. Distinguishes:
- Models that reason well in English but collapse in Swahili.
- Models whose multilingual training preserves reasoning across languages.

---

## How it works as an LLM eval

### Format
- Input: math word problem in language L.
- Output: solution + final integer answer.
- Grading: exact match on the integer.

### Scoring conventions

- **0-shot CoT** with language-native reasoning: the model reasons in the same language as the problem. Simpler for instruct-tuned models.
- **5-shot CoT**: few-shot demonstrations in the same language.
- **Cross-lingual CoT**: reason in English even when the problem is in another language. Often scores higher for English-centric models, but measures a different thing (English reasoning with translation, not native-language reasoning).
- Reported numbers usually average across the 10 non-English languages.

### Typical harness

- lm-eval-harness `mgsm` task.
- Original repo: https://github.com/google-research/url-nlp.

---

## Why it matters

- **Canonical multilingual reasoning eval.** Most tech reports that care about multilingual capability cite MGSM.
- **Tests transfer, not just multilingual knowledge.** Math reasoning is a specific cognitive skill; seeing it work in 10 languages is stronger evidence than "the model can answer facts in 10 languages."
- **Resource-stratified.** Swahili and Telugu are low-resource languages — accuracy on them often lags English by 20+ points, exposing data-scarcity effects.
- **Still actively reported.** 2024–2025 tech reports (Llama 3, Qwen3, Gemini) all include MGSM as a standard column.

---

## Gotchas & tricks

- **Per-language breakdown matters more than the average.** A 91% average can hide 95% on high-resource languages and 75% on Swahili/Telugu.
- **Cross-lingual CoT vs native CoT.** The two give different numbers. Explicit about which is being reported.
- **Translation quality varies.** Professional translators, but occasional ambiguities in math word problems don't translate perfectly. Minor but measurable noise.
- **Approaching saturation on high-resource languages.** German, French, Chinese at 95%+ for frontier models. Low-resource (Swahili, Telugu) still 70–85%.
- **Integer answers are cleanly graded.** Unlike MATH-500 which has LaTeX equivalence issues, MGSM's integer answers are exact-match simple.
- **Doesn't test non-math multilingual reasoning.** MGSM is math-specific. For broader multilingual capability: MMLU-X, Global MMLU, multilingual BIG-Bench.
- **Contamination.** GSM8K has been public since 2021; multilingual translations since 2022. Scores are partially contaminated.

---

## Typical modern numbers (averaged across 10 non-English languages, 0-shot CoT)

| Model | MGSM |
|---|---|
| Claude 3.5 Sonnet | 91.6% |
| GPT-4o | 90.5% |
| Llama 3.1 405B | 91.6% |
| Llama 3.1 70B | 86.9% |
| GPT-4 (0125) | 85.9% |
| Llama 3.1 8B | 68.9% |
| Qwen2.5-72B-Instruct | — |
| DeepSeek V3 | — |

Llama 3.1 405B ties Claude 3.5 Sonnet at the top — one of the strongest multilingual reasoning results at release.

---

## Sources

- Paper: *Language Models are Multilingual Chain-of-Thought Reasoners* — Shi, Suzgun, Freitag, Wang, Srivats, Vosoughi, Chung, Tay, Ruder, Zhou, Das, Wei — Google Research, ICLR 2023, arXiv 2210.03057.
- Paper: *Training Verifiers to Solve Math Word Problems (GSM8K)* — Cobbe et al., 2021, arXiv 2110.14168 — the source of the English problems.
- Repo: https://github.com/google-research/url-nlp — official MGSM release.
- Paper: *The Llama 3 Herd of Models* — 2024 — Llama 3.1 405B reports MGSM 91.6%.
