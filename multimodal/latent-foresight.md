# Latent Foresight

*Depth — condense the future predicted by a frozen video generator into a small set of learnable latent tokens.*

**TL;DR:** VLA policies benefit from "world model" priors — knowledge of what actions cause in the world. Learning that in pixel space from scratch is expensive; existing pretrained video generators already know it. Latent foresight distills that knowledge: attach a small set of learnable **foresight tokens** to the policy, condition them on the current observation and instruction, and supervise them against features from a *frozen* pretrained video generator's forecast of the near future. The tokens condense the video generator's world knowledge into a compact latent code that conditions the action head — no pixel decoding at train time, video branch discarded at inference. Introduced in InternVLA-A1.5 (2026).

**Prereqs:** [README.md](README.md)
**Related:** [masked-boundary-modeling.md](masked-boundary-modeling.md) · [perceptual-flow-matching.md](perceptual-flow-matching.md)

---

## What it is

Standard VLA architectures pair a pretrained VLM (semantic understanding) with an action head (continuous control). To act well, the policy also needs a *dynamics prior*: what happens next given this action. Three options:

- **Learn dynamics from scratch in pixel space.** Expensive, requires massive robot data, tends to erode the pretrained VLM's semantics under multi-objective training.
- **Explicit video-generation branch.** Predict pixels; use the loss as auxiliary. Expensive at inference and still fights the VLM head for capacity.
- **Latent foresight.** Query a frozen pretrained video generator for its features on the near-future forecast; distill those features into a handful of learnable latent tokens that live inside the policy's forward pass.

Only option 3 keeps the pretrained VLM untouched, avoids pixel decoding at train and inference, and inherits world knowledge from the frozen video model for free.

## How it works

Architecturally, the policy contains three streams sharing the same backbone:

1. **VLM head** — continues training on VQA + subtask-prediction tasks, keeping the pretrained backbone semantically alive.
2. **Action head** — a lightweight unified expert that produces continuous actions from the pooled backbone features conditioned on foresight tokens.
3. **Foresight tokens** — a small learnable set $\{f_1, \ldots, f_K\}$ (K on the order of 8–32) that attend into the backbone and are supervised against a frozen video generator's features.

Training signal for the foresight tokens:

```
video_features = FrozenVideoGen.encode( future(obs, action) )   # not decoded to pixels
foresight_features = policy_backbone.query(f_1..f_K, obs, instruction)
L_foresight = || foresight_features - proj(video_features) ||²
```

Where `future(obs, action)` is a short simulated rollout the frozen video generator predicts. The projection matches dimensionalities. Because the video generator is never unfrozen and never decoded, the compute cost is a single forward per training example.

At **inference**, the video-generator branch is discarded. The foresight tokens are computed from the observation alone (the video generator's contribution has already been baked into the backbone through training) and passed to the action head. Inference latency is the bare policy head.

## Why it matters

- **Cheap dynamics inheritance.** Pretrained video generators encapsulate massive world knowledge (physical plausibility, object permanence, contact reasoning). Latent foresight taps into that without paying pixel-generation cost.
- **VLM preservation.** Semantic head keeps training in parallel; the pretrained backbone doesn't drift under action-only objectives. Compositional generalization on held-out instructions improves.
- **Deployable at real-time control.** No pixel decoding at inference; the foresight tokens are ~KB of extra state on the forward pass.
- **General pattern.** The template — condense a frozen generative teacher's forecast into learnable latent tokens — generalizes beyond robotics: any policy that needs dynamics priors from a pretrained temporal model.

## Gotchas & tricks

- **Choose the video generator carefully.** Its priors become the policy's priors. A generator trained on stylized YouTube clips gives the policy stylized-YouTube dynamics.
- **Number of foresight tokens.** Too few and the video generator's features can't be adequately reconstructed. Too many and the tokens dominate the backbone. K in 8–32 is the reported sweet spot.
- **Freeze the video generator hard.** Any gradient into the generator collapses the whole scheme — the point is to inherit its priors, not fine-tune them out.
- **Feature target, not pixel target.** Supervising foresight tokens against reconstructed pixels loses the compression benefit. Match the generator's latent features.
- **Balance losses.** VLM + action + foresight all share one backbone. Uneven loss weighting causes one head to dominate; the InternVLA paper reports careful mixing schedules.

## Sources

- Paper: *InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization* — Ma et al., 2026 — introduces latent foresight for VLAs.
