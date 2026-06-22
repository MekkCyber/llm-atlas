# Image-Editing World Action Models
*Depth — repurposing pretrained image editing models as dynamics models for robot policies.*

**TL;DR:** A **World Action Model** (WAM) predicts what the world looks like after an action. Standard WAMs use **video generation** as the predictor — expensive, capacity-wasteful, and prone to long-horizon drift. **ImageWAM** (Shanghai Jiao Tong U., Tencent Robotics X, Tsinghua, et al., 2026) argues you only need a **single-frame counterfactual edit**: an **image editing model** takes the current observation + an action and produces the next observation. Cheaper inference, fewer action-irrelevant distractors, no error accumulation.

**Prereqs:** [README.md](README.md)
**Related:** [persistent-state-world-models.md](persistent-state-world-models.md), [../evaluation/wrbench.md](../evaluation/wrbench.md)

---

## What it is

The dominant WAM design predicts a **video sequence** $\hat{V}_{t+1:t+H}$ given the current observation $o_t$ and an action $a_t$. The policy uses this prediction to plan. Three coupled limitations:

- **Dense multi-frame futures are expensive** to generate at inference.
- **Capacity is spent on action-irrelevant detail** — lighting, background, camera shake — that don't affect the policy.
- **Long-horizon imagination drifts** — errors compound across frames, misleading the action selection.

ImageWAM swaps the video decoder for a **conditional image editor**: given `(o_t, a_t)`, predict only the **next observation** $\hat{o}_{t+1}$. Multi-step planning chains one-step edits as needed. The image editor inherits from large pretrained image-edit models (InstructPix2Pix-style), bypassing video-pretraining altogether.

## How it works

### Inputs and outputs

```
input:   current observation o_t (image) + action a_t (text or vector)
output:  next observation o_{t+1} (image)
```

Action conditioning can be **text-described** ("pick up the red block") or **continuous** (delta-pose vector projected into the image-editor's conditioning space).

### Reusing pretrained image-edit models

ImageWAM starts from a pretrained text-conditioned image editor and **fine-tunes it on robot-trajectory data** as `(prev_frame, action_caption, next_frame)` tuples. The conditioning is action, not edit-instruction, but the architecture (cross-attention from text tokens to image latents) is unchanged.

### Use in a robot policy

The robot policy (a separate VLA-style head) consults ImageWAM at planning time:

1. For each candidate action $a^{(i)}$, query ImageWAM to imagine $\hat{o}_{t+1}^{(i)}$.
2. Score $\hat{o}_{t+1}^{(i)}$ against a goal image / reward (CLIP similarity, learned scorer).
3. Pick the highest-scoring action.

For longer horizons, recurse: take the chosen $\hat{o}_{t+1}$, expand it as the new $o_t$. The single-step design contains error accumulation because each prediction can be re-grounded against the *actual* observation when available.

## Why it matters

- Inference is **substantially cheaper** than video-generation WAMs — single forward pass through an image editor vs $H$-step video decode.
- Avoids the **long-horizon hallucination** failure mode of video WAMs (see [../evaluation/wrbench.md](../evaluation/wrbench.md), which shows video WAMs fail badly at persistent state).
- Aligns with the broader observation: most "what does the world look like after this action" tasks only need a single-frame counterfactual, not a movie.
- Reuses public image-edit foundation models, dodging the cost of video-WAM pretraining.

## Gotchas & tricks

- **Per-step edit must be accurate.** With single-step prediction the policy bets the next plan on one frame; small visual errors compound through planning even if not through generation.
- **Action conditioning is delicate.** Text-described actions are easier to integrate but lose fine-grained control; continuous vectors need a custom conditioning head.
- **Re-grounding is free and should be used.** Whenever the real $o_{t+1}$ arrives from the robot sensor, use it instead of the predicted one for the next planning step.
- **Doesn't fix persistent-state problems.** ImageWAM is one-step — it inherits the same "what-memory" gap as video WAMs over multi-step horizons. Combine with explicit scene state if persistence matters.
- **Domain gap matters.** Image-edit foundation models trained on web images don't natively handle robot cameras; the fine-tuning step is load-bearing.

## Sources

- Paper: *ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?* — SJTU, Eastern Institute of Technology, Tencent Robotics X, Tsinghua, Zhongguancun Academy, 2026, arXiv 2606.19531.
- Related paper: *Current World Models Lack a Persistent State Core* — 2026, arXiv 2606.20545 — the persistent-state critique of video WAMs that motivates ImageWAM.
- Background: *InstructPix2Pix* and successor image-edit foundation models.
