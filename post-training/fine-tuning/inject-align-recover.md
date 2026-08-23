# Inject, Align, Recover (IAR)
*Depth — three-stage post-training for retrieval-free document knowledge internalization, with model merging as the general-capability recovery step.*

**TL;DR:** To answer questions about a bounded document corpus **without retrieval at inference time**, LLMs must internalize the corpus into weights. Naive continued pretraining injects knowledge but craters general capability; SFT alone doesn't teach the source facts. **IAR** decomposes the problem into three stages: **Inject** (three complementary corpus-conversion objectives), **Align** (answer-only QA SFT), **Recover** (model merge with base instruct to restore general skills). Across Common Corpus / CCI and Llama / Phi / Qwen / SmolLM families, IAR beats Vanilla SFT on all four reported metrics in **7 of 8** settings, averaging **+3.6 pp** domain-QA accuracy and **+12.1 pp** on mean IFEval / MMLU / MSBench.

**Prereqs:** [../../pre-training/mid-training.md](../../pre-training/mid-training.md), [../../pre-training/model-souping.md](../../pre-training/model-souping.md)
**Related:** [../_post-training.md](../_post-training.md), [../rejection-sampling.md](../rejection-sampling.md)

---

## What it is

A staged post-training recipe for **retrieval-free** QA over a fixed corpus:

- **Inject** — mid-training-flavored objective that converts source documents into three complementary loss channels (below).
- **Align** — supervised fine-tuning on answer-only QA pairs (no retrieved context), forcing the model to answer from injected weights.
- **Recover** — merge the injected+aligned model with the untouched base instruct model to restore general skills lost during Inject.

## How it works

**Inject** uses three per-document objectives, run jointly:

1. **Continuation** — next-token prediction on the raw document.
2. **Rewrite** — the model rewrites the source (compression, paraphrase, or expansion) conditioned on the original.
3. **Instruction-conditioned reconstruction** — given a natural-language instruction that names/queries the document, reconstruct the relevant content.

The three objectives together drill the corpus into the model in multiple representational forms — surface strings, paraphrastic forms, and instruction-grounded retrieval-like forms.

**Align.** Standard SFT on `(question, answer)` pairs where the answer is derivable from injected content. Crucially, no retrieved context appears at inference — the model must answer from parameters alone.

**Recover.** Weight-space merge (souping-style) between the Aligned model and the original base instruct model. Merge coefficients trade off domain-QA accuracy against general benchmarks; the paper reports settings on the frontier.

## Why it matters

"Retrieve or fine-tune?" has been dominated by RAG for years, because internalization was too destructive to general capability. IAR's contribution is not any single stage — it's the **staged decomposition + explicit merge-based recovery**, which returns weight-only knowledge to viability for bounded corpora. That matters for on-device deployments, offline agents, and IP-restricted settings where the corpus can't leave a boundary at inference.

## Gotchas & tricks

- LoRA and FAPM can win *individual* general metrics but sit off the domain-vs-general Pareto frontier IAR reaches — a reminder that "domain-adapted at what general cost" is a two-axis question, not one.
- Merge coefficients are hyperparameters — a bad merge either loses domain knowledge (base pulls too hard) or keeps general regressions (aligned pulls too hard).
- Inject's three objectives interact with tokenizer quirks: continuation and rewrite generate different exposure to the source vocabulary; corpora with heavy formatting need custom pre-processing.

## Sources

- Paper: *Inject, Align, Recover: Staged Post-Training for Retrieval-Free Document Knowledge Internalization* — Kou, Shi, Qiu, Zhou, 2026 — [arXiv:2608.20281](https://arxiv.org/abs/2608.20281)
