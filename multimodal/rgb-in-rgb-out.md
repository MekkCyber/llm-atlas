# RGB In, RGB Out (RINO)
*Depth — a unified vision paradigm where every task is expressed as RGB-to-RGB image editing on a single generic backbone.*

**TL;DR:** Language models are general because everything is tokens. Vision models aren't — each task has its own head, output space, and adapter. RINO forces every visual signal (masks, depth maps, keypoints, generation targets) into an RGB image, and every task into an RGB→RGB transformation on a single generic image-editing backbone. Zero-shot, no task-specific fine-tuning, competitive on both dense understanding (segmentation, depth) and dense-conditioned generation (pose-to-image).

**Prereqs:** *(none)*
**Related:** *(no other multimodal depth files yet)*

---

## What it is

A paradigm for unified vision modeling. Two ideas:

1. **Every visual output is RGB.** Segmentation masks become colored images. Depth maps become grayscale RGB. Keypoints become sparse RGB. There is no task-specific output space.
2. **Every task is image editing.** Segmentation: "edit this image so each object is colored uniquely." Depth: "edit this image so brighter pixels are closer." Pose-to-image: "edit this sparse-RGB pose into a photo of a person doing that pose."

The backbone is a **generic image-editing model** (diffusion-based, following the same interface as InstructPix2Pix / consumer image-edit models). No task-specific fine-tuning; task specification lives entirely in the input pair (source RGB + prompt).

## How it works

- **Encode.** Both understanding (segmentation, depth) and generation tasks encode their input as an RGB image.
- **Prompt.** A natural-language instruction specifies the transformation. "Highlight the cat in red." "Show a depth map." "Generate a photo matching this pose."
- **Decode.** The output is another RGB image. Segmentation masks are re-parsed from colored output; depth maps are read from grayscale channels.

The trick is that image-editing backbones trained at internet scale on paired before/after images already know how to make many of these transformations happen. RINO exploits this pretrained capability instead of re-learning each task separately.

## Why it matters

- **Collapses the vision-model zoo.** If unified RGB in/out holds, today's stack of DETR-heads, SegFormer-heads, DPT-heads, ControlNet-adapters collapses into one backbone with prompts.
- **Analogous to text-in/text-out for NLP.** The generality of GPT came from having one interface for all tasks. Vision has been missing that.
- **Zero-shot task transfer.** Adding a new task means writing a new prompt, not fine-tuning a new head. Scales cheaply to long-tail vision tasks.
- **Enables joint scaling.** All vision tasks contribute to and benefit from the same backbone scaling curve, instead of each task's SOTA needing its own scaling regime.

## Gotchas & tricks

- **Precision-critical tasks lose resolution.** Encoding a segmentation mask as an image means output precision is bounded by image resolution and codec quality. Fine-grained masks may need higher output resolution.
- **RGB is lossy for high-dim outputs.** 3-channel RGB has 24 bits per pixel; depth or normals often need more. Trade-off: encode more channels via multi-image outputs, or accept quantization noise.
- **Instruction ambiguity.** "Show the mask" isn't as precise as a class-index output; prompts must be engineered carefully per task.
- **Doesn't beat specialized SOTAs.** Zero-shot RINO is *competitive*, not state-of-the-art. The generality-vs-peak-accuracy tradeoff is real. But general-purpose is often what you want, especially for long-tail tasks where no specialized model exists.
- **Depends on strong image-editing pretraining.** The gains come from what the backbone learned at internet scale. A weak editor produces a weak RINO.

## Sources

- Paper: *Let RGB Be the Language of Vision* — Yang et al., JHU / UCSC / CMU / Rice, 2026 — arXiv 2607.12450.
- Code: https://github.com/yangtiming/RINO.
