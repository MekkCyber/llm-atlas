# Surface-Normal Conditioning for Video Diffusion
*Depth — condition a video diffusion model on reconstructed surface-normal maps instead of aligned reference frames.*

**TL;DR:** Subject-consistent video generation (orbit the camera around a subject while keeping identity stable) normally requires either a dense multi-view training set with frame-level alignment, or a 3D asset. OmniShowcase (Kim et al., FLIPTION / KAIST AI, 2026) proposes a cheaper middle path: reconstruct a **surface-normal map** from a handful of input views, feed it to a video diffusion model as conditioning, and let the diffusion prior fill in texture and lighting. The result is 360° subject-consistent video from a few photos, no 3D asset required.

**Prereqs:** *(none)*
**Related:** [vla.md](vla.md), [semantic-image-tokenizer.md](semantic-image-tokenizer.md)

---

## What it is

A conditioning signal for a video diffusion model that encodes **geometry**, not **pixels**:

- Input: a small set of reference views `{I_1, …, I_K}` of a subject.
- A sparse-view reconstruction stage produces a dense surface-normal map (and/or depth map) covering the full 360° camera trajectory.
- The video diffusion model is conditioned on this normal map, frame-by-frame, in addition to the usual text / reference-image prompt.

The pipeline avoids frame-level pixel alignment entirely. There is no attempt to warp input views into the output trajectory — the diffusion model handles texture and lighting, the normal map handles shape.

---

## How it works

### Data pipeline (training)

```
3D asset library ──► render orbit videos at varied camera paths
                     + ground-truth normal maps per frame
                 ──► train pairs: (normal map sequence, video clip)
```

Entirely synthetic. The authors describe a fully automated rendering pipeline over 3D asset libraries that generates arbitrary orbit paths plus the matched normal-map ground truth.

### Model (training)

A video diffusion model (DiT-style) is fine-tuned to predict video clips conditioned on:

- a text prompt (subject description),
- a sparse-view reference set,
- a per-frame surface-normal map.

The normal map enters as an extra conditioning channel (adapter / ControlNet-style).

### Inference pipeline

```
k reference photos ──► sparse-view normal reconstruction ──► normal-map sequence
                                                                      │
                                                                      ▼
                                                       video diffusion → output clip
```

The sparse-view reconstructor is a smaller, dedicated module trained to go from a handful of views to a dense normal map along a specified camera trajectory.

---

## Why it matters

- **Removes the frame-alignment bottleneck.** Prior subject-consistent video methods needed either per-frame correspondences (expensive, brittle) or a 3D mesh (expensive to acquire).
- **Decouples shape from appearance.** The normal map is a lightweight, geometry-only intermediate. The diffusion prior supplies photorealism, lighting, and motion. The two can be improved independently.
- **Synthetic-only training transfers.** Because the conditioning is geometric, the sim-to-real gap is much smaller than for pixel-conditioned methods — rendered normal maps and reconstructed normal maps live in the same representation.
- **Reusable pattern.** The "condition the generative model on a cheap geometric intermediate" recipe generalizes to depth, segmentation, or part-labels — see ControlNet lineage.

---

## Gotchas & tricks

- **Normal reconstruction quality is the ceiling.** If the sparse-view reconstructor produces smeared normals, the diffusion output inherits the smearing (as blurry/wrong geometry). Reconstruction and diffusion must be trained together or carefully co-tuned.
- **Not well-suited to highly textured non-rigid subjects.** Hair, cloth, fur — surfaces where normals are ill-defined — are handled with lower fidelity. The paper's demos emphasize rigid or near-rigid subjects.
- **Full 360° requires visible-from-any-angle priors.** If the input views cover only the front hemisphere, the back is a hallucination guided purely by the normal map. Expect texture drift on the unseen side.
- **Compute cost is dominated by the diffusion model.** The normal reconstructor is cheap; the video DiT is the expensive part at inference. Standard video-diffusion inference tricks (step reduction, distillation) apply.

---

## Sources

- Paper: *OmniShowcase: Omnidirectional Subject-Consistent Video Generation from Multi-View Images* — Kim, Kim, Kim, Jang, Choo, FLIPTION / KAIST AI / Inha University, 2026, [arXiv 2610.35684](https://arxiv.org/abs/2610.35684).
- Related: ControlNet (Zhang 2023) — the "condition on a cheap geometric map" recipe for image diffusion.
- Related: Zero-1-to-3 / Zero123++ — few-view 3D generation precursors.
