# Context compression via continuous memory tokens
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Long histories and documents are usually compressed either as human-readable text (LLM summarization) or as rendered images (screenshot + OCR). **LatentPress** (Zhou & Sang, 2026) writes them directly into *continuous memory tokens* that a frozen decoder consumes through its input-embedding interface, with no text or OCR intermediate. A small writer adapter (0.1% of decoder params) compresses 4–16×, beats text and OCR baselines on LongMemEval, and is roughly an order of magnitude faster than both.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [kv-cache-eviction](kv-cache-eviction.md)

---

## What it is

A frozen decoder consumes an input as a sequence of embeddings; nothing in its interface requires that those embeddings come from tokenizing text. LatentPress trains a small writer that maps a long conversation or document into a much shorter sequence of **continuous vectors in the decoder's input-embedding space**. The decoder reads them directly — no de-quantization, no text reconstruction, no OCR pass.

Contrast with the two prior compressed-context channels:
- **Text summarization** — writer is an LLM, the intermediate is human-readable text, the reader tokenizes back to embeddings. Loses signal at every hop.
- **Screenshot + OCR** — writer renders text to an image, the reader OCRs it back. Common for vision-language models with long context; slow at both ends.

## How it works

Two roles:

- **Writer.** A small model (4.2M–26.2M trainable parameters) that outputs a sequence of continuous vectors of the *reader's* input-embedding dimension. It sees the raw source (chat history, document) and emits *k* memory tokens where *k* is much smaller than the source length.
- **Reader.** The target LLM, fully frozen. Its input embeddings are the concatenation of the query tokenized as usual and the memory tokens injected verbatim.

Training only updates the writer; the reader stays untouched. The training signal is standard next-token cross-entropy on the reader over supervision pairs where the source has been compressed and the answer must be produced from the memory tokens plus the query.

Compression ratios of 4×, 8×, and 16× are trained by controlling the writer's output length as a fraction of source length. Two zero-shot transfer settings are tested — chat corpus → memory QA, and memory QA → unseen document domains — to isolate the "writer generalizes across content" question.

## Why it matters

**Machine-facing context.** If soft-token memory generalizes across decoders (LatentPress shows two transfer regimes), it becomes a third practical channel alongside text summaries and OCR — and beats both on speed and accuracy at typical compression ratios. That collapses several compressed-context patterns in RAG and multi-turn assistant stacks.

**Frozen reader.** The decoder is untouched, so the compression is deployable without a fine-tune — matters for closed-weight or already-serving models where you can only preprocess the input.

## Gotchas & tricks

- **Per-decoder writer.** Because memory tokens live in a specific decoder's embedding space, the writer is decoder-specific by default. Zero-shot cross-decoder transfer isn't the setting tested.
- **16× is a real cliff.** On LongBench-QA, 16× compression trails raw context; 4–8× matches or beats. Don't push compression past 8× without a task-specific eval.
- **Writer size vs. reader size.** 0.1% of decoder params is enough for the tested settings; the paper doesn't establish whether that generalizes as decoders grow.
- **The interface is fragile to prompt template changes.** Injecting memory tokens breaks if the reader's chat template inserts control tokens between memory and query; be explicit about the placement.

## Sources

- Paper: *LatentPress: Context Compression Beyond Text and Vision* — Zhou & Sang, LinkedIn, 2026 — [arXiv:2609.01507](https://arxiv.org/abs/2609.01507).
