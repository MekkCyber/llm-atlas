# Low-resource language adaptation of a RAG stack
*Depth — end-to-end recipe for adapting a modern retrieval-augmented-generation pipeline to a target language, grounded in the Nemotron-Greek case.*

**TL;DR:** Adapting a modern RAG stack to a low-resource language is not "swap the tokenizer and continue" — each component (embedder, reranker, reader) needs its own target-language supervision. The Nemotron-Greek recipe walks all four stages: mine a domain-mixed corpus, generate synthetic retrieval pairs, fine-tune embedder + cross-encoder reranker, and LoRA-tune a large-MoE reader for grounded generation.

**Prereqs:** [../data/_data-curation.md](../data/_data-curation.md), [../data/quality-filtering.md](../data/quality-filtering.md)
**Related:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)

---

## What it is

A staged adaptation recipe applied to NVIDIA's Nemotron retrieval stack for Modern Greek. Also a datapoint that a strong parameter-free baseline (BM25) can outperform off-the-shelf multilingual dense models on specialist target-language text — so target-language finetuning is not optional if you want dense-retriever gains to hold on domain data.

## How it works

**Stages.**

1. **Corpus mining.** Assemble a domain mix (legal, energy, financial, medical) from web + specialist sources in the target language. Apply standard quality filters.
2. **Synthetic supervision.** Generate `(query, positive, negative)` retrieval pairs — the Nemotron-Greek run produced 65,773 pairs — with a strong multilingual generator, then filter for consistency.
3. **Embedder fine-tuning.** Fine-tune a 1B embedder on the synthetic pairs. Typical wins are large (nDCG@10 0.362 → 0.835 in the Greek case).
4. **Reranker adaptation.** Cross-encoder reranker fine-tuned on the same pairs; consistent per-domain gains.
5. **Reader fine-tuning.** LoRA-tune the reader on grounded-generation examples — the recipe used a 30B-A3B MoE reader, judged answer correctness 29.4% → 66.9%.

**Evaluation.** Domain-mix retrieval scores + a released target-language RAG benchmark (HERA in the Greek case). Both intra-domain and general-domain transfer are measured; the *BM25 gap* is domain-dependent even after adaptation.

## Why it matters

Two takeaways for anyone deploying RAG in a low-resource language:

- **BM25 is a real baseline.** Off-the-shelf multilingual dense models are not automatically better on specialist target-language corpora; the dense win comes from *target-language finetuning*, not the base model.
- **Component-level adaptation composes.** Improving only the embedder yields a bounded downstream lift; the reader-side LoRA tune is what turns retrieval gains into answer-quality gains.

Also: adaptation transfers from the specialist mix to general-domain target-language text, at least for the embedder — a useful headroom argument for corpus mixing.

## Gotchas & tricks

- Synthetic pair quality drives everything downstream — filter aggressively.
- Reader-side LoRA on a large MoE requires careful expert routing behavior on target-language tokens; monitor router entropy on target-language batches.
- Publish a target-language benchmark alongside the model — otherwise cross-team comparisons stall.

## Sources

- Paper: *Teaching Nemotron Greek: Mining a Corpus, Adapting Retrieval, and Grounding Generation for Modern Greek across Specialist Domains* — Kirouane, Petrocheilos, et al., 2026 — [arXiv:2608.05138](https://arxiv.org/abs/2608.05138)
