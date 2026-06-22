# WRBench
*Depth — diagnostic benchmark for persistent state in video world models.*

**TL;DR:** A diagnostic benchmark that probes whether a video generation model maintains **world state** when the camera looks away and comes back. Camera trajectories are designed to (1) observe an initial scene, (2) leave view while events change the scene, (3) re-observe, and check whether the model renders the *correct end state* — not just a visually plausible one. 23 models × 9,600 videos. Identifies a **preservation–access–re-observed-consistency gap** that aggregate video metrics (FVD, FID, CLIPScore) miss entirely. Lu et al., 2026.

**Prereqs:** *(none)*
**Related:** [../multimodal/README.md](../multimodal/README.md), [fid.md](fid.md)

---

## What it is

Existing video-generation benchmarks score **visible fidelity** — does each rendered frame look correct? WRBench asks a sharper, world-model-shaped question: **does the model's internal state evolve correctly while the camera looks elsewhere, so that re-observation is consistent?**

A WRBench probe has three phases:

1. **Observe.** Camera shows scene with named, trackable entities (object position, state, count).
2. **Leave & mutate.** Camera moves away from the entities; described or implied events change them off-camera (a cup is filled, a box opens, an object moves).
3. **Re-observe.** Camera returns to the original viewpoint. The benchmark checks whether the re-rendered scene reflects the post-mutation state.

The benchmark decomposes capability into three sub-skills:

| Sub-skill | What it tests |
| --- | --- |
| **Preservation** | Is the unobserved event encoded in latent state at all? |
| **Access** | Can the model retrieve that state when needed? |
| **Re-observed consistency** | Does re-observation produce the right end state regardless of camera path? |

## How it works

### Probe construction

WRBench draws scenes from a controlled set of templates (kitchen, desk, shelf) with synthetic and real-video sources. For each scene, a camera-trajectory generator produces multiple paths that leave and return to the same vantage. The mutation is specified by text and verified by ground-truth renders.

### Grading

Each generated video is graded by:

- **Visible fidelity** (baseline FVD / per-frame CLIPScore).
- **State match** at re-observation, scored by an evaluator VLM and verified by template rules. Two failure modes are isolated:
  - **In-place state change** (cup gets filled in the same location) — hardest, because object presence gives no hint.
  - **Relocation** (cup moves to another spot) — easier, because the absence-and-re-presence is itself a hint.

### Headline metric

`re-observed-state correctness`, reported separately from visible fidelity. The paper documents a large gap between the two across 23 models.

## Why it matters

- **Aggregate video benchmarks don't measure world-model capability.** A model can ace FVD and CLIPScore while violating persistence.
- **Distinguishes "renderer" from "world model."** Models that render motion gracefully but can't keep state are exposed as renderers with no internal world.
- Motivates **endpoint-persistence training objectives** and explicit **"what-memory"** mechanisms as research priorities — and lends weight to alternative WAM designs (see [../multimodal/image-editing-wam.md](../multimodal/image-editing-wam.md)) that sidestep video prediction entirely.

## Gotchas & tricks

- **In-place vs relocation gap is the diagnostic signal.** A model that's strong on relocations but weak on in-place changes is exploiting object-presence cues, not modeling state.
- **Evaluator VLM matters.** State checks are LLM-graded; calibration across grader VLMs is a known sensitivity. The paper reports inter-grader agreement.
- **Synthetic-vs-real distribution matters.** Models trained heavily on synthetic camera trajectories may overfit to the WRBench probe distribution.
- **Camera-path coverage is partial.** Re-observation can be from the original vantage, an adjacent angle, or a fully novel angle; the paper splits results by path family.

## Sources

- Paper: *Current World Models Lack a Persistent State Core* — Lu, Zhu, Shi, Cai, Tang, Chen, Cao, Tang, Zhang, Dai, Ju (USTC, X-Humanoid, CAS, PKU), 2026, arXiv 2606.20545.
- Related concept: persistent-state and "what-memory" requirements for video world models — discussed in the paper's framing.
