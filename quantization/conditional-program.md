# Conditional Program (2-Bit N-Gram LUT Quantisation)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A compressed format for the **n-gram lookup tables** that back conditional-memory LMs. Each row of the table is stored as a short code plus a local per-row scale, and a shared **Gaussian trellis decoder** reconstructs only the rows a given input requests. At 2 bits per weight, 4×–8× smaller than full-precision tables, and 41–63% lower output KL divergence than 2-bit round-to-nearest, with most task scores near the original. On DeepSeek, table storage drops from 203 GB to 51 GB — freeing GPU memory for larger serving workloads.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** [attention](../fundamentals/attention.md)

---

## What it is

A growing family of LMs condition their forward pass on an external *n-gram lookup table*: at each step the model computes a short context hash and reads out a learned vector from a huge table (billions of rows). These tables are a non-trivial fraction of total deployment memory — on DeepSeek-scale models, up to 200+ GB per replica.

Naive 2-bit quantisation of these tables (RTN or GPTQ-style per-row) *works for weights but breaks for LUTs*: LUT rows are row-sparse and distributionally wider than typical attention weights, so round-to-nearest loses accuracy fast. Conditional Program (CP) is a format designed for this access pattern.

## How it works

**Row representation.** Each row $r$ is stored as:

- A **short code** $c_r$ of $b$ bits (e.g. $b=2$).
- A **local scale** $s_r$ in a few bits (e.g. 4–8).

**Trellis decoder.** A single, model-wide **Gaussian trellis** $D$ maps $(c_r, s_r)$ → reconstructed row $\hat{W}_r$ by stepping through a Viterbi-style path parameterised by a shared Gaussian prior over the row-vector distribution. Because $D$ is small and shared, its storage is amortised across all rows.

**Row-sparse decode.** At inference, the model emits a context hash, looks up only the few rows that hash hits, and runs $D$ on each. No full-table dequantisation is ever materialised. This is exactly the row-sparse access pattern the format is designed for — unlike weight matrices, which are accessed densely.

## Why it matters

- **Specific to LUT access.** Weight-quantisation methods assume dense access per layer; LUT quantisation needs row-sparse access, which CP's trellis decoder gives for free.
- **Better than RTN at the same bitrate.** 41–63% lower output KL at 2 bits across DeepSeek / Qwen / LongCat. The win concentrates at low bitrates where RTN collapses.
- **Changes the serving stack maths.** The 203 GB → 51 GB drop on DeepSeek frees GPU memory that can be re-spent on larger batches, longer context, or on-GPU KV cache rather than off-GPU swap.
- **Composes with weight quantisation.** CP is orthogonal to FP8 / INT4 weight schemes — attack them together.

## Gotchas & tricks

- **Trellis decoder is model-specific.** The Gaussian prior is fit to a given model's row distribution. Transferring a trellis across models costs a few percent KL.
- **Local scale resolution matters.** Dropping the scale below 4 bits visibly hurts KL. 4–6 bits per row is the sweet spot — still contributes little to total size because rows outnumber scales.
- **Hash collisions amplify quantisation error.** Multiple contexts mapping to the same row inherit the same quantisation noise. Pair with hash-salt rotation during calibration.
- **Not a drop-in for every LUT.** Rows that are far from the Gaussian prior (highly spiky distributions) see smaller gains over RTN. Measure row-wise KL before committing.
- **No fine-tuning required.** CP is calibration-only; no gradient steps on the main model. Calibration data: a few thousand token blocks suffice.

## Sources

- Paper: *N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models* — anonymous, 2026 — [arXiv:2610.21810](https://arxiv.org/abs/2610.21810).
- Related: *N-Gram Models as Conditional Memory for LLMs* — prior work establishing the LUT-conditioned LM primitive.
