# Role-Coupled Tokenization (SpecSlot)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In pixel-space generative transformers (JiT family) the patch grid couples three roles — *reading* input evidence, *writing* output pixels, and *communicating* globally — so modeling finer detail blows up global attention cost. **SpecSlot** decouples these roles via a three-stage pipeline: *Read* evidence at a sparse anchor grid, *Allocate* each anchor into K slots that own disjoint radial-DCT frequency bands, *Route* only the lowest-frequency slot through global attention while all slots mix locally. On ImageNet-512, SpecSlot-K4 improves FID by 2.84 over a FLOPs-matched JiT.

**Prereqs:** [transformer-block.md](transformer-block.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Classical pixel-space generative transformers (JiT, VAR, MAR) tokenize images on a single spatial grid, and every token is simultaneously (a) a reader of input pixels, (b) a writer of output pixels, and (c) a participant in the global attention sequence. Making detail finer means more tokens means longer global sequence means quadratic blowup. SpecSlot breaks the coupling: a sparse anchor grid handles global attention, and each anchor is subdivided into *slots* that handle detail via a frequency decomposition.

## How it works

**Read.** The input is pooled into a sparse grid of spatial *anchors* (coarse). Each anchor absorbs evidence from its spatial neighborhood.

**Allocate.** Each anchor is split into $K$ slot states. The slot index $k$ owns a disjoint radial-DCT (discrete cosine) frequency band — slot 0 owns the DC-and-lowest band, slot $K-1$ owns the highest. The slot states are the parameters that will carry fine detail.

**Route.** Only *slot 0* from each anchor enters the global attention sequence; all slots at the same anchor mix via local attention. Global sequence length is thus the number of anchors — independent of $K$. A pixel-level spatial head reconstructs output pixels from the full stack of slot states at each anchor's local neighborhood.

Increasing $K$ adds detail capacity *without* lengthening the global sequence. The frequency-band basis ensures slots represent complementary information rather than redundant copies.

## Why it matters

The patch-grid-as-everything design is the main cost barrier for high-resolution pixel-space generation. Role-coupled tokenization decouples "where to attend" from "what detail to represent" — the global attention budget scales with anchor count, not with resolution. The ImageNet-512 result (+2.84 FID over FLOPs-matched JiT) and the text-to-image transfer (SpecSlot beats JiT-B/8 on DPG-Bench at 40.6% lower forward cost) both suggest the decomposition generalizes beyond class-conditional ImageNet.

## Gotchas & tricks

- **Radial DCT is non-obvious.** A plain learned subdivision would work less well; the paper's ablations show the frequency structure is what makes slots non-redundant.
- **Slot 0 is doing double duty** — it participates in global attention *and* owns the low-frequency band. If you want to swap in another basis, keep this property.
- **Local attention within an anchor is dense across slots.** Only the global sequence is sparsified.
- **Not a hierarchical U-Net.** The anchor grid stays flat; slots are per-anchor frequency bands, not resolution levels.
- **The transfer to T2I uses the same anchor density** as class-conditional — the compute savings come from K, not from reducing anchors.

## Sources

- Paper: *Read, Allocate, Route: Role-Coupled Tokenization for Pixel-Space Generative Transformers* — (authors not shown on HF page), 2026 — [arXiv:2610.40647](https://arxiv.org/abs/2610.40647).
- Reference: *JiT* and *MAR* — the pixel-space generative-transformer baselines.
