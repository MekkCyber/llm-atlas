# Multi-Byte Prediction (MBP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Byte-level hierarchical LMs (MegaByte, BLT, SpaceByte) decode one byte at a time — a throughput cliff. **Multi-Byte Prediction** generalises [multi-token prediction](mtp.md) to bytes with a **variable-length prediction window aligned to the hierarchical LM's latent segments**, plus a custom attention mask that keeps causality intact. No new parameters; Pareto-optimal quality/throughput trade-off across QA, instruction-following, summarisation, and MT.

**Prereqs:** [mtp.md](mtp.md), [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [../fundamentals/bpe.md](../fundamentals/bpe.md)

---

## What it is

Byte-level hierarchical LMs run a *latent* transformer over segments (dynamically-computed patches of bytes) and a *byte* transformer that decodes the bytes of the current segment. Decoding is autoregressive at the byte level: one byte, then look at the byte, then the next byte, and so on.

Multi-Byte Prediction (MBP) predicts *all bytes in the current latent segment in parallel*, so a segment of $s$ bytes costs one forward instead of $s$. Because segment length is dynamic and data-dependent, the prediction window has to be dynamic too.

## How it works

### Dynamic prediction window

At position $t$ the hierarchical encoder emits a latent token $z_j$ that covers bytes $t{+}1 \ldots t{+}s_j$. MBP asks the byte decoder to emit all $s_j$ bytes at once, conditioned on $z_j$ and prior bytes only. When the next segment is short (e.g. a single-character token), the window is 1; when long (e.g. a URL or a numeric literal), the window is large. This tracks the *natural* boundaries of the segmenter rather than a fixed $k$.

### Causality-preserving attention mask

If byte $b_i$ inside the segment attends to byte $b_{i+1}$, causality breaks. MBP uses a **staircase mask** inside the window:

```
prediction of  b1 : attends to  z_j, prior_bytes
prediction of  b2 : attends to  z_j, prior_bytes, b1
prediction of  b3 : attends to  z_j, prior_bytes, b1, b2
...
```

So within one parallel forward, byte $b_i$ can see all earlier bytes in the same segment but not later ones — the ordering is preserved even though the whole window is emitted in one pass. All $s_j$ byte losses accumulate to the training objective.

### No extra parameters

MBP re-uses the existing byte decoder's LM head; only the attention mask changes. This distinguishes it from MTP's approach of attaching a new prediction module per extra step.

## Why it matters

- **Removes the byte-level throughput cliff.** A segment of average length ~4–6 bytes becomes ~4–6× cheaper to decode at negligible quality cost.
- **Keeps tokeniser-free training.** Byte-level LMs sidestep BPE lock-in — better robustness to typos, multilingual coverage, code — but were bottlenecked by decode speed. MBP restores throughput without reintroducing subword tokenisation.
- **Composes with speculative decoding.** The parallel-byte draft can itself be a speculative-decoding draft head against a larger verifier.

Reported result: MBP is Pareto-optimal on the quality × throughput frontier across QA, instruction-following, summarisation, and MT — matches next-byte quality with substantially higher tokens/second.

## Gotchas & tricks

- **Segment quality dictates MBP quality.** A weak segmenter (one that emits very short or very jagged segments) undercuts the parallelism gain.
- **Loss weighting inside the window.** Later positions in a segment condition on more predicted bytes and are typically easier — reweighting the loss toward the first position keeps the training signal balanced.
- **Watch decoder capacity.** Because the byte decoder now must model up to $s_{\max}$ steps in one forward, it needs enough width to internalise longer intra-segment structure than pure next-byte training required.
- **Not the same as MTP.** MTP adds prediction modules for fixed *token* offsets $k=1,\ldots,D$ on a subword LM. MBP has zero extra parameters and predicts a *variable* number of bytes within a segment that is *itself* chosen by the segmenter.
- **Inference infrastructure.** The KV cache and the mask kernel must both handle the window shape — an off-the-shelf decode loop will regress to next-byte if not adapted.

## Sources

- Paper: *Dynamic Multi-Byte Prediction With Hierarchical Language Models* — Owodunni, Okocha, Grant, Limisiewicz, Kumar — Ohio State / Oklahoma / ETH Zurich, 2026 — https://arxiv.org/abs/2608.15454
- Prior art: *Better & Faster LLMs via Multi-token Prediction* — Gloeckle et al., 2024 — the subword-level ancestor. Byte-level LMs: MegaByte (Yu et al., 2023), Byte-Latent Transformer (Meta, 2024), SpaceByte (Slagle, 2024).
