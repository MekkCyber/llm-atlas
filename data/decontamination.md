# Decontamination
*Depth — removing training documents that overlap with evaluation benchmarks.*

**TL;DR:** If eval questions (or their answers) appear verbatim in the training corpus, benchmark scores are inflated in ways that don't generalize. Decontamination is the pipeline step that detects and removes training documents overlapping with known evaluation sets. The standard approach since GPT-3 is n-gram-overlap filtering: for each eval example, remove training documents sharing an exact n-gram of length ≥ L (common choices: L = 13 for GPT-3; L = 8–50 in later work). Easy to implement, imperfect in practice, but meaningfully reduces score inflation when applied conservatively.

**Prereqs:** [tokenization](../fundamentals/_tokenization.md)
**Related:** [deduplication](deduplication.md), [quality-filtering](quality-filtering.md), [dolma](dolma.md), [_data-curation](_data-curation.md), [olmo-2 case study](../case-studies/olmo-2.md)

---

## What it is

Pretraining corpora are scraped from the web. Public evaluation benchmarks (MMLU, GSM8K, HumanEval, etc.) are also on the web, often quoted verbatim on blogs, tutorials, and answer sites. Uncontrolled, a training corpus may contain:

- The benchmark's exact questions (copy-paste from the paper).
- Question + answer pairs (tutorial solutions, course websites).
- Question + full worked solution (study-guide style).

A model exposed to these during pretraining can memorize them and "solve" the benchmark without having learned the underlying capability. Reported scores then overestimate real-world performance.

Decontamination is the step that filters training documents overlapping with eval examples, *before* pretraining begins.

## How it works

### N-gram overlap filtering — the GPT-3 recipe

GPT-3 (Brown et al., 2020) established the standard approach:

1. For each benchmark example, extract its text (question + options, or prompt).
2. Tokenize and shingle into n-grams of length `N` (GPT-3 used `N = 13`).
3. Build a set of all such n-grams across all eval benchmarks → the "contamination set".
4. For each training document, check if any of its n-grams hit the contamination set.
5. Remove (or flag and truncate) training documents with hits.

Implementation: a Bloom filter or hash set of eval n-grams lets you check each training document in O(doc length).

### Choice of N

`N` controls sensitivity vs. specificity:

- Small `N` (say 5): catches partial quotes but fires on common phrases ("the capital of France is"), over-removing legitimate training data.
- Large `N` (say 50): catches only long verbatim copies, misses paraphrased leaks.
- `N ≈ 13` (GPT-3 default) is the empirical sweet spot for English prose.

Different labs tune this differently. Llama 3 reports using 8-gram overlap with an additional overlap-ratio threshold. OLMo 2 decontaminates against a curated eval list using a multi-length n-gram match. The exact parameters matter less than running *something* consistently against your full eval suite.

### What counts as an "eval set"

Decontamination is only as thorough as the list of benchmarks checked. Standard practice covers:

- Multiple-choice QA: MMLU, ARC, HellaSwag, PIQA, Winogrande.
- Math: GSM8K, MATH.
- Code: HumanEval, MBPP.
- Reading comp: BoolQ, RACE.
- Long-form: various.

Public benchmarks are a moving target — new ones arrive, old ones get derivative test sets. Decontamination lists need updating when new evals enter the community consensus.

### When decontamination fails

N-gram overlap has well-known blind spots:

- **Paraphrased leaks.** A tutorial rewriting GSM8K problems in its own words shares no long n-grams with the benchmark but leaks the same content. N-gram filters don't catch this.
- **Translated leaks.** Benchmark content translated into other languages appears in multilingual corpora; n-gram matching in the original language misses it.
- **Answer-only leaks.** The model may see the answers without the questions (e.g. a study-guide listing "Q17: 42"). Without the question text, n-gram filters can't tell it's benchmark-related.
- **Structural leaks.** Some benchmarks have templated formats ("Q: ... A: ..."). Training data with the same template pattern but different content can prime the model on the format, inflating scores without direct leakage.

The practical consequence: decontamination is a *mitigation*, not a guarantee. Modern practice (Llama 3, OLMo 2) pairs it with held-out private evaluations that weren't public during pretraining to cross-check reported scores.

## Why it matters

- **Correct benchmark reporting.** The difference between contaminated and decontaminated scores on some benchmarks is several points. Without decontamination, reported numbers mislead downstream model choosers.
- **Fairness across models.** If one lab decontaminates aggressively and another doesn't, their reported scores aren't comparable. Decontamination is a coordination good.
- **Prerequisite for [mid-training](../pre-training/mid-training.md) data curation.** Mid-training mixes are heavily weighted toward math/code/reasoning content — which maps exactly onto common benchmarks. Without decontamination at this stage, the leverage described in the mid-training file becomes benchmark-hacking.

## Gotchas & tricks

- **Decontaminate once per final eval list.** Running decontamination against an incomplete list and later discovering the model was trained on a leaked benchmark is a "redo the whole run" problem. Keep the eval list maximal from the start.
- **Normalize before hashing n-grams.** Lowercase, strip punctuation, collapse whitespace. "What is 2+2?" and "What is 2 + 2 ?" should produce the same n-grams.
- **Check for contamination post-hoc.** Even with upstream decontamination, compute contamination statistics after training completes (for the released training-data snapshot vs. the final eval set). This catches mistakes and is increasingly expected in responsible reporting.
- **Remove, don't mask.** Some pipelines replace matching n-grams with `[REMOVED]` tokens rather than dropping the document. This is cheaper but can still leak the eval *format* to the model via the surrounding text.
- **Public vs. private evals.** Decontaminate against all *public* evals. Private held-out evals are your insurance: even if upstream decontamination has holes, private eval scores approximate real-world capability.
- **Post-training decontamination is different.** SFT and RL datasets should also be decontaminated against eval benchmarks. This is a separate pass — don't assume pretraining-time decontamination covered your SFT mix.
- **Don't over-decontaminate code.** Code benchmarks (HumanEval, MBPP) have generic stub functions ("def solution(x):") that match legitimate training code. Length-based filtering before contamination check avoids removing unrelated short snippets.

## Sources

- Paper: *Language Models are Few-Shot Learners* (GPT-3) — Brown et al., 2020 — the 13-gram decontamination recipe that became standard.
- Paper: *Data Contamination: From Memorization to Exploitation* — Magar & Schwartz, 2022 — shows how memorization vs. generalization contamination both happen.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — details its decontamination pipeline including n-gram parameters and eval coverage.
- Paper: *2 OLMo 2 Furious* — AI2, 2024 — decontaminates OLMo-Mix and Dolmino against eval list.
- Paper: *Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research* — Soldaini et al., 2024 — documents the decontamination pass in Dolma.
- Paper: *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples* — Yang et al., 2023 — empirical study of paraphrased-leak blind spots in n-gram decontamination.
