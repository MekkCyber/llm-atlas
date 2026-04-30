# GPQA
*Depth — "Google-proof" graduate-level science questions.*

**TL;DR:** **448 multiple-choice questions** in **biology, physics, and chemistry** at the PhD/postdoc level. Written by domain experts, with a **"Google-proof" constraint**: non-expert humans with internet access get only ~34% (vs 65% expert accuracy). Designed to resist both memorization (contamination) and web-search-based gaming. The **GPQA Diamond** subset (198 questions) is the most-cited "hard reasoning" eval for frontier models. Introduced by Rein et al. (2023, arXiv 2311.12022). Current frontier reasoning models (o1, R1) score 70–75%; non-reasoning frontier ~50–60%.

**Prereqs:** *(none)*
**Related:** [mmlu-pro](mmlu-pro.md) · [aime](aime.md) · [math500](math500.md)

---

## What it is

Rein et al., *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*, arXiv 2311.12022.

- **448 total questions** across biology, physics, chemistry.
- **Multiple choice, 4 options**.
- **Three subsets of increasing difficulty**:
  - **GPQA** (full): all 448 questions.
  - **GPQA Main** (refined, 448 → some subset via filtering): validated expert-answerable.
  - **GPQA Diamond** (198 questions): "hardest subset" — where two expert validators independently agreed on the correct answer AND where non-experts consistently failed.
- The "Diamond" subset is the most-reported number — more than the full set.

### Google-proof constraint

Each question designed so that **non-expert humans with full internet access** (≥30 minutes per question) score no better than ~34%. Experts (PhD-level in the subject domain) score ~65%. The question design forces:
- Domain expertise beyond undergraduate.
- Multi-step reasoning across concepts.
- Resistance to keyword search (answers cannot be found by searching the question text).

This is enforced by having paid annotators (PhD candidates and experts) try to solve with web access; questions that non-experts could answer were rejected.

### Topics

- **Biology**: cell biology, genetics, molecular biology, biochemistry, evolutionary biology.
- **Physics**: classical mechanics, quantum mechanics, thermodynamics, electromagnetism, astrophysics.
- **Chemistry**: organic, inorganic, physical, analytical, biochemistry.

Question-length distribution: typically 50–200 words. Some include diagrams (released as image attachments separately).

---

## How it works as an LLM eval

### Format
- Input: question + 4 answer choices A–D.
- Output: letter.
- Grading: exact match.

### Scoring conventions

- **0-shot CoT**: the paper's default. Reasoning models are expected to CoT through the problem.
- **Pass@1** at `T = 0` is the standard; `avg@N` for noisy reasoning models.
- **GPQA Diamond** (198 questions) is the most-quoted subset.

### Typical harness

- Simple-evals (OpenAI).
- lm-eval-harness `gpqa` task.
- Original repo: https://github.com/idavidrein/gpqa.

---

## Why it matters

- **Contamination-hard by design.** Google-proof construction means the web (the source of most pretraining data) doesn't directly contain the answers. Scores are closer to genuine capability.
- **Graduate-level reasoning frontier.** The 448 problems span bio / phys / chem at PhD depth — a level where "pattern-matching from pretraining" becomes much weaker than reasoning.
- **Discriminates reasoning models from non-reasoning.** Non-reasoning frontier (GPT-4o, Claude 3.5 Sonnet) lands 50–60%; reasoning models (o1, R1) jump to 70–75%. Clean separation.
- **Less gameable than MMLU.** MMLU's contamination and shallow-question problems don't apply.

---

## Gotchas & tricks

- **Diamond vs full GPQA.** Papers sometimes cite "GPQA 51%" (full) or "GPQA Diamond 59%" (subset). The same model's GPQA vs Diamond scores can differ by 5–10 points. Check which.
- **Expert ceiling is ~65%.** Models exceeding ~70% Diamond are already at / beyond expert level — it's a meaningful saturation point, not an arbitrary one.
- **CoT is essential.** Without CoT, even frontier models regress significantly. For reasoning-trained models with implicit CoT, no special prompt needed.
- **Some questions require diagrams.** The base text-only eval skips or substitutes text descriptions for diagram-dependent questions. Multimodal evals handle them natively.
- **Small N → high variance.** 198 Diamond questions → one question ≈ 0.5 pp. Pass@1 at T > 0 needs averaging.
- **Domain imbalance.** Physics is the hardest subset; biology tends to saturate first. Per-domain breakdowns matter for diagnosing where a model's reasoning falls short.
- **Public — contamination creep.** Released Nov 2023. By now in some training data. But the Google-proof design makes memorization still hard — if the model saw the question, it may have also seen the question without an answer.

---

## Typical modern numbers (GPQA Diamond, Pass@1 / 0-shot CoT)

| Model | GPQA Diamond |
|---|---|
| o1 (full) | 77.3% |
| o3 | >85% |
| DeepSeek-R1 | 71.5% |
| Claude 3.5 Sonnet | 59.4% |
| GPT-4o | 53.6% |
| Llama 3.1 405B | 51.1% |
| Kimi k1.5 | (not reported) |
| Llama 3.1 70B | 46.7% |
| Llama 3.1 8B | 32.8% |
| Human expert | ~65% |
| Non-expert with internet | ~34% |
| Random baseline | 25% |

---

## Sources

- Paper: *GPQA: A Graduate-Level Google-Proof Q&A Benchmark* — Rein et al., 2023, arXiv 2311.12022.
- Repo: https://github.com/idavidrein/gpqa.
- Paper: *The Llama 3 Herd of Models* — 2024 — reports GPQA for all three Llama 3 sizes.
- Paper: *DeepSeek-R1* — 2025 — GPQA Diamond 71.5%, explicitly trained against GPQA-class problems.
