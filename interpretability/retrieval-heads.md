# Retrieval Heads

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A sparse subset of attention heads (typically <5% of all heads) is responsible for *locating* evidence in long inputs — the mechanism behind needle-in-a-haystack success. These "retrieval heads" are identifiable, model-agnostic, largely stable across modalities, and surgically trainable: a ranking-objective fine-tune of just these heads adds 6.03 points to Qwen3-VL-8B's MMDocIR Recall@1.

**Prereqs:** [../architectures/multi-head-attention](../architectures/multi-head-attention.md), [../architectures/mla](../architectures/mla.md)
**Related:** [../fundamentals/attention](../fundamentals/attention.md)

---

## What it is

A circuit-level finding: among the hundreds of attention heads in a modern LLM, a *sparse* set does most of the work of fetching information from the context — e.g. attending from a question token back to the exact answer span. These heads light up on needle-tests and darken on summarization tasks. They are preserved across multimodal fine-tunes: the same heads retrieve from text, image patches, and audio features.

## How it works

### Identification

Wu et al. (2024) and the broader "retrieval head" literature use a per-head copy score on needle-in-a-haystack data: for each (question, document, needle) triple, compute how strongly head $h$ attends from the generated answer token back to the needle span. Average over many triples; rank heads by score. The top-$k$ heads (usually $k \ll$ total heads) are the retrieval heads.

### Cross-modal persistence (Li et al. 2026)

Running this identification on 18 checkpoints — base LLMs, VLMs, audio-visual LMs — the sparse retrieval-head set is:

- **Largely shared** across modalities (same head indices, same layers).
- **Mostly inherited** from the text backbone through multimodal fine-tuning.
- **Causally necessary** for audio-visual retrieval: ablating them removes essentially all of it.

### Targeted training

Apply a ranking objective directly to the retrieval-head attention logits — e.g. maximize the softmax of the correct evidence span under those heads' attention patterns — leaving the rest of the model frozen. Qwen3-VL-8B gains +6.03 pts MMDocIR Recall@1, far above general fine-tuning.

## Why it matters

- Localizes a capability (long-context retrieval, cross-modal grounding) to a small set of parameters, enabling cheap, surgical fine-tunes.
- Explains why multimodal fine-tunes often inherit text retrieval ability without re-learning: the heads that implement retrieval don't need to be rebuilt per modality.
- A reusable interpretability result: retrieval heads are among the most robustly-reproduced circuit findings in modern LLMs, and now extend to MLLMs.

## Gotchas & tricks

- Retrieval-head identification depends on the needle probe; shift the probe (longer context, different distractors) and the set shifts slightly at the edges.
- Ablating retrieval heads also hurts in-context learning tasks that rely on retrieval; don't ablate for a quick capability test unless you accept that collateral damage.
- "Shared across modalities" is a majority statement, not universal — some audio-specific retrieval heads do appear in audio-visual fine-tunes.
- Head-targeted ranking training can overfit to the probe distribution; mix in general SFT data to keep broad capabilities.

## Sources

- Paper: *A Broader Look into MLLM's Retrieval Heads* — Li, Li, Lin, Yu, IIE-CAS / Tencent, 2026 — [arXiv:2610.32411](https://arxiv.org/abs/2610.32411).
- Prior: *Retrieval Head Mechanistically Explains Long-Context Factuality* — Wu et al., 2024 — the original text-only retrieval-head identification.
