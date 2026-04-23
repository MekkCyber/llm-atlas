# IFEval — Instruction-Following Evaluation
*Depth — verifiable-instruction compliance, scored programmatically.*

**TL;DR:** ~**541 prompts** containing **verifiable instructions** ("write in more than 400 words", "end with phrase X", "use exactly N bullet points", "output valid JSON"). 25 instruction types across 9 categories. Grading is **programmatic checks on the response string** — no LLM judge. Four metrics: **prompt-level and instruction-level accuracy, each in strict and loose variants**. Introduced by Zhou et al. (Google, Nov 2023). The standard "does the model follow structural instructions" eval — now part of Open LLM Leaderboard v2, Hugging Face's eval suite, and OpenAI simple-evals. Frontier models land 85–92% at prompt-strict; not quite saturated.

**Prereqs:** *(none)*
**Related:** [mmlu](mmlu.md) · [_post-training](../post-training/_post-training.md)

---

## What it is

Paper: *Instruction-Following Evaluation for Large Language Models*, Zhou, Lu, Mishra, Brahma, Basu, Luan, Zhou, Hou, Google, 2023, arXiv 2311.07911.

- **~541 prompts**, each containing 1–3 verifiable instructions.
- **25 instruction types**, grouped into **9 categories** (Table 1):
  - **Keywords** — forced words, forbidden words, letter-frequency constraints.
  - **Language** — response language (Spanish, Korean, etc.).
  - **Length Constraints** — word count ranges, sentence count, paragraph count.
  - **Detectable Content** — "include a postscript", "include a title".
  - **Detectable Format** — valid JSON, exactly N bullet points, specific section headers.
  - **Combination** — multiple stacked constraints.
  - **Change Cases** — all caps / all lowercase / specific case patterns.
  - **Start/End** — response must begin/end with specific text.
  - **Punctuation** — no commas / specific markers.
- Each instruction is **verifiable by a deterministic Python function** (the repo ships one per instruction type).

Repo: https://github.com/google-research/google-research/tree/master/instruction_following_eval.

---

## How it works as an LLM eval

### The four metrics

Zhou et al. Section 2.2 + 3. For each response:

- **Prompt-level accuracy** — 1 if *all* verifiable instructions in the prompt pass, else 0.
- **Instruction-level accuracy** — fraction of individual verifiable instructions that pass.

And each in two variants:

- **Strict** — check the response as generated.
- **Loose** — check the response after 8 robustness transforms (strip markdown `**`/`*`, strip first line, strip last line, and combinations). Pass if *any* transform passes.

So four reported numbers:

1. **Prompt-strict** — "all instructions in prompt pass, on the raw response".
2. **Prompt-loose** — "all instructions in prompt pass, on some transform of the response".
3. **Instruction-strict** — "fraction of individual instructions that pass on raw response".
4. **Instruction-loose** — "fraction of individual instructions that pass on some transform".

Loose metrics reduce false negatives from cosmetic formatting. Strict is harder and more commonly reported as the headline.

### Grading implementation

For each instruction type, a Python function checks the property. Example (length):

```python
def check_num_words(response, min_words, max_words):
    n = len(response.split())
    return min_words <= n <= max_words
```

Deterministic; no ambiguity.

### Input / output

- **Input**: a natural-language prompt containing one or more verifiable instructions.
- **Output**: any natural-language response.
- **Grading**: the Python check functions run on the response string.

### Typical reporting

Headline number is usually **Prompt-strict accuracy**. Some papers report all four; some only report "IFEval" without specifying which metric (assume prompt-strict unless stated).

---

## Why it matters

- **The only widely-used post-training eval for structural instruction-following.** Most benchmarks measure knowledge, reasoning, or preference alignment. IFEval specifically measures "if I ask for 5 bullet points, do I get 5 bullet points." This is a real production concern that other benchmarks don't catch.
- **Programmatic grading.** No LLM judge, no human judge, no subjectivity. Reproducible and cheap. A complete evaluation runs in minutes.
- **Not yet saturated at the strict level.** Frontier models hit ~85–92% Prompt-strict; ceiling is ~95–97%. Still discriminates top models. Instruction-strict saturates earlier (90–95% at the frontier).
- **Correlates with real-world instruction-following pain.** Models that fail IFEval's length/format/language constraints tend to fail similar constraints in production. The eval is directly actionable.
- **Part of Open LLM Leaderboard v2.** Hugging Face adopted IFEval as one of the six benchmarks for the v2 leaderboard, which means every open model gets an IFEval score by default.

---

## Gotchas & tricks

- **Verifies surface form, not spirit.** "Include a postscript" checks for the string "P.S.". A model that produces a functionally postscript-like paragraph without the literal "P.S." fails. For structural instructions this is fine; for semantic instructions IFEval is a shallow proxy.
- **English-centric.** Most prompts and instructions are English. The language-constraint category includes other languages but they're a small fraction. Non-English instruction-following is underspecified by IFEval.
- **Easy for models that pattern-match instructions.** A model fine-tuned specifically on IFEval-style data will over-score. Contamination risk is real — the prompts and instruction types are public.
- **Strict vs loose gap is typically 3–8 pp.** A Prompt-strict / Prompt-loose gap > 10 pp suggests the model emits cosmetic formatting (asterisks, leading spaces) that the strict check penalizes but loose strips. This is recoverable with a post-processing step.
- **Prompt-level vs instruction-level is load-bearing.** A prompt with 3 instructions gets credit in instruction-level even if only 2 pass, but scores 0 in prompt-level. Papers that report only instruction-level can hide systematic failures on compound instructions.
- **Doesn't measure harder compositional instruction-following.** Multi-turn constraints, long-horizon tasks, reasoning-gated instructions ("if the user asks for X, include Y") — all out of scope. IFEval is *structural*, not *behavioral*.
- **Doesn't capture ambiguous or conflicting instructions.** Real users frequently give contradictory constraints; IFEval's prompts are clean. The benchmark underrepresents this failure mode.
- **Prompts are sometimes unnatural.** Several IFEval prompts stack 3 strict constraints in ways a real user wouldn't. This makes the benchmark harder than natural use, which is both a feature (headroom) and a bug (distribution mismatch).
- **GPT-4 vs Claude vs open models' prompt-strict numbers cluster 80–88.** Small reporting gaps in this range can flip with minor prompt formatting changes. Don't over-interpret sub-3-pp differences.

---

## Typical modern numbers (IFEval, prompt-strict / instruction-strict)

| Model | Prompt-strict | Instruction-strict |
|---|---|---|
| GPT-4 / GPT-4o | ~80–88% | ~86–92% |
| Claude 3.5 Sonnet | ~86% | ~92% |
| Kimi k1.5-short | 87.2% | — |
| DeepSeek-V3 | 86.1% | — |
| Llama-3.1-405B-Instruct | 86.0% | — |
| Llama-3.3-70B-Instruct | 74–80% | 83–87% |
| Qwen2.5-72B-Instruct | 84.1% | — |
| 7–8B open instruct models | 55–70% | 65–78% |

---

## Sources

- Paper: *Instruction-Following Evaluation for Large Language Models* — Zhou, Lu, Mishra, Brahma, Basu, Luan, Zhou, Hou, Google, 2023, arXiv 2311.07911 — the original IFEval paper. Table 1 lists the 25 instruction types; Section 2.2 + 3 define the metrics.
- Repo: https://github.com/google-research/google-research/tree/master/instruction_following_eval — reference implementation.
- Hugging Face Open LLM Leaderboard v2: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard — uses IFEval as one of six benchmarks.
- OpenAI `simple-evals`: https://github.com/openai/simple-evals — includes IFEval.
- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI, 2025, arXiv 2501.12599 — reports IFEval Prompt-strict 87.2%.
