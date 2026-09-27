# Spatial Audio Encoders for Audio-LMs (FOA Path)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Add a dedicated **first-order ambisonics (FOA) spatial branch** alongside an existing semantic audio encoder, then fuse the two before passing them to an LLM. This lets omni-modal models localise sound direction — "where did that come from?" — while keeping the existing "what is that sound?" pathway untouched. Enables joint spatial-audio + visual reasoning for embodied agents.

**Prereqs:** [README.md](README.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md), [../agents/README.md](../agents/README.md)

---

## What it is

Off-the-shelf audio-LM encoders (Whisper, EnCodec-based front-ends, Qwen-Audio's semantic path) are trained on mono or stereo and represent *content*: what words, what event, what timbre. They discard spatial information: where the source is in the room relative to the listener.

The **spatial audio encoder** adds a parallel branch that ingests **first-order ambisonics** (FOA) signals — a 4-channel representation (W, X, Y, Z) that mathematically encodes sound direction — and produces per-frame direction-of-arrival features. Fused with the semantic path, the resulting model can answer both semantic and spatial questions in one context.

## How it works

Front-end:

- **Semantic branch.** Reuses a pretrained audio-LM encoder (e.g. Qwen-Audio) unchanged; produces content tokens.
- **Spatial branch.** A small transformer trained on FOA-labeled audio to output DOA-aware features. Training data: audio clips with known source azimuth/elevation, augmented via HRTF-based re-panning.
- **Fusion.** Per-frame concatenation of semantic and spatial features, projected into the LLM's token space via a shared adapter.

Downstream: the audio-LM is co-trained on tasks that require *both* modalities — sound localisation, sound-guided navigation, spatial audio-visual QA — using standard next-token cross-entropy plus task-specific auxiliary heads (DOA regression for the spatial task).

The OmniEcho paper packages this with OmniEchoBench, a 6-task / 197-scene benchmark that isolates spatial audio-visual perception from purely visual or purely semantic evaluation.

## Why it matters

- **Ears agents can actually use.** Most audio-LMs treat spatial cues as noise; embodied agents in real environments (nav, VR, robotics) need direction-of-arrival. This is the first-generation architectural fix.
- **Additive, not disruptive.** The pattern preserves the existing semantic path — no retraining of the base audio encoder — and only bolts on a second branch. Cheap to adopt for existing audio-LM stacks.
- **Non-vision multimodal push.** Multimodal LLMs are heavily vision-biased; the spatial-audio recipe is a rare gain along the audio axis and opens up sound-source-aware agent behaviours.

## Gotchas & tricks

- **FOA availability is limited.** Most large-scale audio datasets are mono/stereo; the spatial branch needs FOA or a stereo-to-FOA lifter (which introduces errors near the median plane).
- **HRTF-based augmentation is essential.** Without it, the spatial branch overfits to the recording setup used for training and fails on new rooms/microphones.
- **Fusion granularity matters.** Frame-level fusion loses direction info in long segments; keep the spatial features at fine temporal resolution and let the LLM downsample.
- **Sync with vision.** For audio-visual tasks, the semantic and spatial branches must be aligned to the same visual frame rate; naive frame-drop skews the DOA task.
- **Distinct from beam-forming.** The spatial encoder learns to *represent* direction; classical beam-formers actively enhance a chosen direction. The two are complementary — beam-forming as a pre-processing step feeding into the spatial encoder gives compounded gains.

## Sources

- Paper: *OmniEcho: Audio-Visual Spatial Understanding for Omni-Modal Embodied Agents* — Liu, Wang, Xie, et al. (Qwen team + consortium), 2026, [arXiv 2609.23407](https://arxiv.org/abs/2609.23407) — introduces the FOA branch alongside a semantic audio-LM path and the OmniEchoBench evaluation.
- Related: *Qwen-Audio* — Alibaba, 2023 — the semantic audio-LM backbone this pattern extends.
