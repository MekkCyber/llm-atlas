# Case Study: SenseNova-U1.5

*An 8B **native unified** multimodal model from SenseTime, September 2026. Understands, reasons about, and generates visual content inside one **encoder-free and VAE-free** stack — no CLIP tower, no VAE latent bridge. Built on a **Mixture-of-Transformers (MoT)** backbone, trained with **spatially-coherent patch reconstruction**, and post-trained by **multi-expert on-policy distillation** across four visual specialists.*

**Related concepts:** [mixture-of-transformers](../multimodal/mixture-of-transformers.md) · [native-unified-models](../multimodal/native-unified-models.md) · [multi-expert-distillation](../post-training/multi-expert-distillation.md) · [../multimodal/image-tokenizers](../multimodal/README.md) · [../architectures/_moe](../architectures/_moe.md) · [../fundamentals/attention](../fundamentals/attention.md)

---

## What this is

SenseNova-U1.5, released September 2026 by SenseTime. An 8B **Mixture-of-Transformers** decoder that handles both visual understanding *and* visual generation without any separate vision encoder (SigLIP/CLIP) and without any pretrained VAE latent bridge for the generation side. The model consumes and emits image patches directly through the same transformer stack that reasons in text.

The paper matters because it commits to the **native unified** thesis at frontier scale: prior "unified" models stitched together a vision encoder, an LLM decoder, and either a diffusion or VAE decoder — three trained systems and two conversion boundaries. SenseNova-U1.5 argues that the conversion boundaries themselves are the problem, and validates the stack-collapse claim with wins across image fidelity, complex composition, multi-reference editing, bilingual text rendering, and interleaved generation.

---

## Architecture at a glance

```
8B-parameter Mixture-of-Transformers (MoT) decoder
  ├─ per-modality experts:  text  /  image (understanding)  /  image (generation)
  ├─ shared global self-attention across all modalities per layer
  ├─ encoder-free input path:    raw image patches → linear embed → MoT
  └─ VAE-free output path:       backbone emits patches / native discrete image codes

Native resolution: up to 4K on the generation side
Modality routing: by modality tag, not by learned gate
```

Details of the MoT structure, expert widths, and layer count are not disclosed in the abstract; see the paper for concrete numbers.

---

## Training recipe

### Pretraining

- **Objective mix.** Standard next-token prediction on interleaved text + image-patch sequences (understanding), joint with **patch reconstruction** on the visual side.
- **Spatially-coherent patch reconstruction.** A dense auxiliary loss where random image patches are masked and the model reconstructs them from surrounding patches. This is what "strengthens the visual interface" in the paper's language — it gives pixel-level supervision without requiring a separate VAE.
- **Generation-side scaling.** Carefully curated generation + editing data, improved task formulations, structural prompt enhancement, and native resolutions up to 4K on the generation targets. The paper emphasizes *data* and *task formulation* as the levers scaling generation quality — not new architectural pieces.

### Post-training

**Four specialist experts trained separately:**
1. Visual aesthetics
2. Bilingual (Chinese + English) text rendering inside generated images
3. Infographic generation
4. Image editing

**Multi-expert on-policy distillation.** The generalist student rolls out on-policy; per-domain rollouts are scored against the corresponding specialist; on-policy KL matches the student to the specialist for that domain. All four specialists' behaviors are consolidated into the shipped 8B model. See [multi-expert-distillation](../post-training/multi-expert-distillation.md).

Exact per-stage hyperparameters, dataset sizes, and compute budget are not given in the abstract.

---

## Key results

Reported wins across:
- **Image fidelity** (unconditional and prompt-conditional generation quality)
- **Text rendering** inside generated images (bilingual: Chinese + English)
- **Complex composition** (many-object, layered prompts)
- **Multi-reference editing** (edit conditioned on multiple reference images)
- **Interleaved generation** (long sequences alternating text and image outputs)
- **Instruction following** improves alongside the generation objective, not against it
- **Subject identity + geometry preservation** on edits — the model changes what the instruction says to change and leaves the rest alone
- **Generalization to long, structured visual instructions** despite limited exposure to such formats in the generation training data

Concrete benchmark scores are in the paper; the abstract highlights the *classes* of tasks improved rather than headline numbers.

---

## What's individually novel from this tech report

The three depth files that ship with this case study are the innovations for which SenseNova-U1.5 is the primary source in the graph:

- **[mixture-of-transformers](../multimodal/mixture-of-transformers.md)** — per-modality expert transformers sharing global attention; modality-tag routing (not learned gate).
- **[native-unified-models](../multimodal/native-unified-models.md)** — the encoder-free + VAE-free design pattern with spatially-coherent patch reconstruction as the pretraining prior.
- **[multi-expert-distillation](../post-training/multi-expert-distillation.md)** — on-policy consolidation of $M$ visual specialists into one generalist policy.

Other pieces used here (Mixture-of-Experts more broadly, standard transformer attention, autoregressive image generation) are covered by existing pages under `architectures/`, `fundamentals/`, and `multimodal/`.

---

## Why this case study matters

Every "unified multimodal" release in the last 18 months has argued about *how to glue* a CLIP-family encoder to a diffusion VAE. SenseNova-U1.5 makes the strong claim that the glue itself is the problem: build the model native, and understanding capacity transfers to generation without a conversion boundary. If it holds up at frontier scale, the multimodal stack collapses from three trained systems to one, and visual planning + visual creation land on the same substrate as text reasoning.

The recipe is also a template beyond vision: MoT + native representations + multi-expert distillation applies unchanged to audio-LLMs and video-LLMs.

## Sources

- Paper: *SenseNova-U1.5: Towards Native Unified Visual Intelligence* — SenseTime (Haiwen Diao, Jiahao Wang, Chenjing Ding, … Dahua Lin, Wenxiu Sun, Ziwei Liu, et al.), 2026 — [arXiv:2609.11929](https://arxiv.org/abs/2609.11929)
- Code / project page: https://github.com/OpenSenseNova/SenseNova-U1
