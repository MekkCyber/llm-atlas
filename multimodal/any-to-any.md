# Any-to-Any Multimodal Models
*Depth — a single network that predicts any modality from any combination of other modalities.*

**TL;DR:** Traditional multimodal models fix an input/output pattern (text-in / image-out, image-in / text-out). **Any-to-any** models take any subset of modalities (text, image, depth, audio, segmentation…) as input and produce any other subset as output — chosen at inference. Historically this required encoder-decoder or diffusion backbones trained from scratch. **Decoder-only any-to-any** models (Modus, 2026) unify all modalities into one token stream so a pretrained LLM decoder can be warm-started, no modality-specific heads, losses, or task pipelines.

**Prereqs:** [README](README.md)
**Related:** [../architectures/README.md](../architectures/README.md)

---

## What it is

A model architecture with:

- **Symmetric modalities.** Every modality is tokenized into the same discrete or continuous token space; no distinguished "input side" and "output side."
- **Arbitrary input/output subsets.** At inference, the caller specifies which modalities they're providing and which they want back. The model conditions on the former and generates the latter.
- **No task-specific heads.** No separate "image head" or "depth head." One backbone, one loss, one training pipeline.

Contrast with:

- Encoder-decoder (e.g., 4M): asymmetric, still needs modality-aware components.
- Diffusion any-to-any (e.g., ImageBind + diffusion decoders): can't reuse strong pretrained LLM decoders.
- VLMs: fixed input pattern (text + image → text).

## How it works

```
tokens = concat(
    modality_tokens[m](x_m)  for m in provided_modalities
)
prompt = tokens + [BOS_out(m)] for m in requested_modalities
completion = decoder.generate(prompt)
outputs = {m: modality_detokens[m](completion.segment(m)) for m in requested_modalities}
```

- Per-modality tokenizers/detokenizers project into and out of the shared token space (usually a VQ-style codebook or a continuous latent embedding).
- The decoder itself is modality-agnostic — a standard causal transformer.
- Modality-selection tokens (`<image>`, `<depth>`, etc.) delimit sections in the sequence, letting the model condition on the current output modality.

## Why it matters

- **Reuses LLM decoders.** Warm-starting from a pretrained text decoder is orders of magnitude cheaper than training a diffusion any-to-any from scratch.
- **Composable at inference.** Adding a new modality means adding a tokenizer + short fine-tune, not redesigning the model.
- **Slots into the LLM stack.** Existing infrastructure — RL, distillation, quantization, serving frameworks — applies without changes.
- **Scientific / cross-domain use.** Ecology, astronomy, biomedical: same architecture across diverse modality mixes.

## Gotchas & tricks

- **Tokenizer quality gates the whole system.** A bad image tokenizer (blocky reconstructions) caps output quality regardless of how good the decoder is.
- **Compute imbalance across modalities.** An image is 256+ tokens; a class label is 1. Naive batching wastes compute on padding. Use packing or modality-aware sequence layout.
- **Modality-specific priors are lost.** A dedicated diffusion image model still beats decoder-only on image quality alone — you trade peak per-modality quality for compositional flexibility.
- **Continuous vs. discrete tokens.** Discrete (VQ) tokens are simple but lossy; continuous latents preserve more detail but need a modified loss.
- **Training data mixture is subtle.** Balancing across (input-subset, output-subset) combinations is a large hyperparameter surface; naive equal weighting biases toward common combos.
- **Evaluation is under-defined.** No single benchmark covers "any-to-any." Reporting typically enumerates the per-task scores of the same weights across many combinations.

## Sources

- Paper: *Modus: Decoder-Only Any-to-Any Modeling of Diverse Modalities* — An et al., 2026 — [arXiv:2607.25948](https://arxiv.org/abs/2607.25948).
- Related: 4M (Bachmann et al.), Chameleon (Meta), ImageBind, Unified-IO 2 — prior any-to-any and unified-multimodal architectures.
