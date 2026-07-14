# Visual Pretraining for Language Intelligence
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of extracting text from documents and pretraining an LLM on the resulting plain text, feed the model the *rendered page image* and let it learn from pixels. Across multiple backbones and benchmarks, visual pretraining on the same underlying corpora consistently beats text-only pretraining — evidence that the "convert everything to text" default of LLM data pipelines is silently discarding information.

**Prereqs:** none.
**Related:** [../data/quality-filtering.md](../data/quality-filtering.md) · [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Every LLM pretraining pipeline runs a text-extraction stage — HTML → Markdown, PDF → text, OCR, table-flattening, math-LaTeX cleanup. This stage is where an enormous amount of *information* is lost: figures become gone, page layouts flatten to newlines, typeset equations become mangled ASCII, tables become CSV or worse.

**Visual pretraining** skips that stage. Rendered page images are fed directly to a foundation model, which learns from pixels — the layout, figures, equations, and typography included. The paper systematically studies several unsupervised visual pretraining paradigms and shows they scale better than text-only pretraining on the *same underlying documents*.

---

## How it works

### Setup

For a document corpus $\mathcal{D}$, two parallel pretraining tracks:

- **Text-only baseline**: apply the standard extraction stack to $\mathcal{D}$, pretrain a language model on the resulting text.
- **Visual pretraining**: render each document in $\mathcal{D}$ to page images, pretrain a foundation model directly on those images (with an unsupervised objective like masked-patch reconstruction, or generative next-patch, or a hybrid text-token target where available).

Same underlying corpus, different input modality.

### The paradigms studied

The paper explores multiple unsupervised visual-pretraining objectives:
- Pure image-side objectives (masked patch prediction).
- Image-plus-latent-text objectives (predicting an aligned text stream from the rendered page).
- Multi-stage curricula that interleave visual and text-conditioned steps.

Details vary by backbone, but the shared finding is that the visual side dominates across the ablation grid.

### Downstream evaluation

The models are evaluated on standard language-understanding benchmarks — same suite one would use for a text-only pretrained LM. The comparison is apples-to-apples in corpus, compute, and evaluation.

---

## Why it matters

- **Text extraction is lossy.** Figures, equations, tables, and layout information routinely carry the actual signal in scientific papers, technical documentation, and web pages. Skipping the extraction stage stops discarding it.
- **Scales.** The paper's central empirical claim is that visual pretraining is not just competitive but *scalable* — the gap over text-only grows (or at least holds) as backbone size and data scale up.
- **Blurs the LLM/VLM line at pretraining time.** Modern VLMs typically bolt vision onto a pretrained text-only LM. This paper argues the "vision-first" starting point may itself be better for *language* intelligence.
- **Simplifies the pipeline.** No OCR, no HTML cleaner, no LaTeX extraction — one rendering step and the model handles the rest.

---

## Gotchas & tricks

- **Compute per token / patch is higher.** Image patches carry less token-level semantic density than words; matched-quality pretraining likely requires more FLOPs per document. The paper's "scalable" claim rests on the *downstream* return, not the pretraining efficiency.
- **Rendering choices matter.** Font size, resolution, page-break policy — the paper's setup uses one particular rendering pipeline; other choices could shift results.
- **Not (yet) shown at frontier scale.** The paper studies visual pretraining across multiple backbones and benchmarks but does not report a full frontier-scale run against a Llama-/DeepSeek-class baseline. The result is directional, and the community will need larger-scale replications.
- **Downstream text-in / text-out use still requires an interface.** A model pretrained on rendered pages still needs to accept text prompts and emit text answers at inference; the paper covers this but the interface layer is a design choice.
- **Interacts with tokenizer choice.** For a text-only baseline, byte- vs subword tokenization affects the comparison. The paper's ablations should be read with the choice of textual baseline in mind.

---

## Sources

- Paper: *Scalable Visual Pretraining for Language Intelligence* — Zhang, Zhao, Zhang, Zhao, Lin, Zhou, Song, Liu, Ye, Huang, Gu, Lv, Guo, Liu, Wang, Chen — [arXiv:2607.09657](https://arxiv.org/abs/2607.09657).
- Related lineage: OCR-free document understanding (Donut, Pix2Struct) — earlier work showing pixel-in / text-out reduces extraction loss on document tasks; the current paper argues the same idea should shape *pretraining itself*.
