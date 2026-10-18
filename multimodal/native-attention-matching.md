# Native Attention Matching
*Depth — classification-via-attention output head for closed-set VLM outputs.*

**TL;DR:** For VLM tasks whose answer space is closed — multiple-choice, image-point selection, numerical binning — autoregressive decoding is overkill. Native attention matching replaces the generation head with a classification head that scores each candidate by attention between the question tokens and the candidate's embedding, in a single forward pass. MJev-Spatial applies this to spatial-reasoning questions on Qwen3.5-4B and reports ~5.6× speedup vs. its autoregressive base at higher accuracy.

**Prereqs:** [multi-head-attention.md](../architectures/multi-head-attention.md), [attention.md](../fundamentals/attention.md)
**Related:** [transformer-block.md](../architectures/transformer-block.md)

---

## What it is

An output-format change, not a backbone change. The VLM encoder is unchanged; what's replaced is the decoder's role for closed-set outputs. Candidates are embedded (as text options, image patches, or numerical bin centers) and the head picks the top-attention candidate.

## How it works

1. For a question with context `c` (image + text) and candidate set `{a_1, …, a_K}`, embed each candidate `a_i` into the same hidden space as `c`.
2. Compute attention scores `s_i = QK^T` where `Q` is the question's summary token and `K_i` is `a_i`'s embedding; apply softmax, pick top.
3. Train with cross-entropy on candidate labels. 12.3M QA pairs reported.
4. Inference = one forward pass, no decoding loop.

## Why it matters

Latency on multiple-choice VQA (counting, pointing, bin-selection) is dominated by autoregressive overhead; a single-pass classifier bypasses it entirely while sitting on top of the same backbone. The pattern generalizes to any closed-set VLM output, not just spatial.

## Gotchas & tricks

- Candidate embedding is the design surface — text options are easy (tokenize); image points need a tokenizer that can embed arbitrary coordinates; numerical values need bin centers (resolution tradeoff).
- Open-ended questions still need generation; a classifier head is a *specialized* output path, not a replacement for the full decoder.
- Backbone must be frozen or jointly fine-tuned; a backbone that only ever saw autoregressive loss will not have calibrated candidate-score magnitudes.

## Sources

- Paper: *MJev-Spatial: System One Spatial Intelligence via Native Attention Matching* — authors not disclosed on HF page, 2026 — [arXiv:2610.22276](https://arxiv.org/abs/2610.22276)
