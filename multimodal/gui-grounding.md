# GUI Grounding
*Depth — mapping a natural-language reference to pixel coordinates in a screenshot.*

**TL;DR:** **GUI grounding** is the sub-task inside every computer-use agent of "turn a textual reference ('the Send button in the top right') into an `(x, y)` click target in the current screenshot." End-to-end VLMs are mediocre at this: screenshots contain hundreds of dense UI elements, text is small, and the training-data distribution underweights desktop UIs. Modern recipes split the task — a smaller **grounding model** proposes a region of interest, the big VLM reasons about it. PreView (Hou & Xu, 2026) is a training-free preprocessor in this family: a lightweight model predicts a rough center, crops and zooms, overlays a red-circle visual cue, and emits a textual description — the frozen downstream VLM then clicks more accurately.

**Prereqs:** [../agents/computer-use-agents.md](../agents/computer-use-agents.md)
**Related:** [../agents/visual-prompting-cue.md](../agents/visual-prompting-cue.md), [vla.md](vla.md)

---

## What it is

Formally, GUI grounding is:

```
input:  (screenshot I, natural-language reference r)
output: pixel coordinate (x, y) in I that corresponds to r
```

It is the perception-to-action bottleneck for every CUA — misclicks are the dominant failure mode on OSWorld-class benchmarks. Three families of approaches:

| Approach | Strength | Weakness |
| --- | --- | --- |
| **End-to-end VLM click prediction** | simple, no extra component | poor on dense UIs, small text |
| **Set-of-marks + VLM** | boost VLM via overlays | overlay must be computed, can clutter |
| **Grounding model + VLM** | strong on dense UIs | extra component, latency |

The third is becoming dominant. The grounding model is small (sub-billion to few-billion params), trained on dense UI-labelled data, and specialized for the "reference → coord" subtask.

---

## How it works (PreView recipe)

PreView decomposes grounding into four lightweight steps that any frozen VLM can benefit from:

```
1. Center prediction.   A tiny trained model predicts an approximate (x, y) for r in I.
2. Local view cropping. Crop a window around the predicted center, zoom in.
3. Visual cue overlay.  Draw a red circle around the candidate, add a textual hint.
4. Downstream grounding. The frozen VLM (Qwen3-VL / OS-Atlas) consumes the enriched
                        input and produces the final click.
```

Only step 1 is trained; steps 2–4 are deterministic. The frozen VLM is untouched.

### Why the red circle helps

Visual markers (red circles, numbered boxes, bounding-box overlays) consistently improve VLM grounding on dense UIs. They act as **attention priors** — the VLM already attends to high-saliency pixels, and the overlay engineers that saliency. See [../agents/visual-prompting-cue.md](../agents/visual-prompting-cue.md).

### Measured lift

Across 5 GUI grounding benchmarks, PreView improves the average accuracy of **Qwen3-VL-2B, OS-Atlas-4B, and Qwen3-VL-4B** — training-free lifts on already-strong baselines. The paper reports the gains per benchmark; exact numbers in Table 2 of the arXiv preprint.

---

## Why it matters

- **Grounding is where CUAs break.** The reasoning is often fine; the click is wrong. Improving grounding has an outsized effect on end-to-end task success.
- **Decouples "where to look" from "where to click."** A small specialized model handles ROI; a large general-purpose VLM handles reasoning. The pattern reuses across agent domains.
- **Training-free deployment.** PreView ships as a preprocessing adapter — no modification to the downstream VLM, no re-finetune. Easy to integrate.
- **Composes with set-of-marks.** Overlay-based grounding methods stack: PreView's red circle + a set-of-marks numbering scheme can be used together.

---

## Gotchas & tricks

- **Center-prediction error is the ceiling.** If the small grounding model picks the wrong element, no amount of overlay-engineering recovers. Train the grounding model on data representative of the target UIs.
- **Crop size tradeoff.** Tight crops lose context ("is this the Send button or Save button?"); loose crops dilute the visual cue. Multi-scale crops help.
- **Overlay style matters.** Red circles work better than translucent boxes on most VLMs; numbered boxes help when multiple candidate elements survive the first pass. Pick empirically.
- **Dense text UIs still underperform.** Spreadsheets, IDEs, dashboards — the pixel density defeats both the grounding model and the VLM. Combine with DOM / AX tree input when available.
- **Not a replacement for a trained CUA backbone.** PreView lifts frozen VLMs by small-to-moderate amounts; a VLM post-trained on CUA trajectories (OS-Atlas, UI-TARS) still wins outright on the top of the leaderboard. The two can be combined.

---

## Sources

- Paper: *PreView: Enhancing GUI Grounding with Preprocessed Views and Cues* — Hou & Xu, 2026, [arXiv 2610.33645](https://arxiv.org/abs/2610.33645).
- Paper: *Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V* — Yang et al., 2023 — the overlay-prompting ancestry.
- Paper: *OS-Atlas: A Foundation Action Model for Generalist GUI Agents* — Wu et al., 2024.
- Paper: *UI-TARS: Pioneering Automated GUI Interaction with Native Agents* — Yan et al., ByteDance, 2025.
