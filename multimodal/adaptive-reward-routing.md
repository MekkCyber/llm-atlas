# Adaptive Reward Routing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For multi-reward RL fine-tuning of joint audio-video diffusion models, Adaptive Reward Routing makes both the *location* of reward-driven updates and the *weighting* across competing rewards adaptive — using cross-attention responses as a proxy for cross-modal influence, and branch-specific reward-gradient interactions as residual corrections to user-chosen weight priors.

**Prereqs:** [../post-training/_rewards.md](../post-training/_rewards.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/dara.md](../post-training/dara.md), [../post-training/corr-grpo.md](../post-training/corr-grpo.md)

---

## What it is

Yang et al. (Tencent, 2026) address a long-standing mess in multi-reward RL for joint-modality generative models: both *where* each reward's gradient should be applied and *how* competing rewards should be combined are usually set by fixed heuristics, so RL fails to track the evolving model function during training. Their method makes both dynamic, inside the DiffusionNFT forward-process RL framework for joint audio-video diffusion.

## How it works

Two components.

**(i) Cross-Modal Influence-Guided Routing (localising updates).**
Bidirectional cross-attention responses between audio and video streams serve as an efficient proxy for how much each cross-modal layer currently matters. Token-aware losses and gradient scales across cross-modal layers are reweighted on the fly using these responses — layers currently carrying strong cross-modal information get larger-magnitude updates, inert layers get smaller. No additional model heads or auxiliary losses.

**(ii) Preference-Preserving Modality-Aware Reweighting (coordinating rewards).**
User-specified reward weights are kept as **priors** rather than hard targets. During a warm-up phase these priors dominate. After warm-up, branch-specific **reward-gradient interactions** (how each reward's gradient aligns with the policy update) are computed and applied as *residual corrections* to the preference weights. The correction only reallocates — it preserves the ordering requested by the user — but prevents dominant rewards from suppressing weaker but essential objectives.

## Why it matters

- Consistent improvements over strong RL baselines on modality quality (per-stream), cross-modal semantic alignment, and audio-video synchronisation.
- First method to tie *both* routing and weighting to measurable, in-training signals (cross-attention magnitudes, reward-gradient alignments) — removes the hand-tuned reward-weight schedule as a hidden hyperparameter.
- Preserves user intent via the preference-prior design — operators can still steer the model, the method just prevents gradient dynamics from silently overriding the chosen weights.
- The pattern generalises to any multi-stream diffusion / flow model where one modality's gradients could swamp another's.

## Gotchas & tricks

- Cross-attention magnitude as an influence proxy assumes the architecture actually uses cross-attention for the inter-modal binding — architectures with alternative fusion (e.g. shared tokens) need a different proxy.
- Warm-up length is a knob: too short and the preference prior never dominates; too long and the method's adaptivity doesn't engage.
- Preserves prior *ordering* of weights, not their exact values — if an operator needs exact per-reward weights, the residual-correction step must be ablated.
- Evaluated on joint audio-video DiT backbones; transfer to video-only or image-only multi-reward RL is suggested but not demonstrated.

## Sources

- Paper: *Adaptive Reward Routing: Dynamic Multi-Reward Optimization for Joint Audio-Video Diffusion via Forward-Process RL* — Yang, Zhao, Zhang, Wang, Li, Liu, Zhao, Rao; Tencent, 2026 — [arXiv:2609.37200](https://arxiv.org/abs/2609.37200).
- Context: *DiffusionNFT* — the forward-process RL framework for diffusion models that Adaptive Reward Routing plugs into.
- Context: GRPO-style multi-reward aggregation work (see [../post-training/dara.md](../post-training/dara.md), [../post-training/corr-grpo.md](../post-training/corr-grpo.md)) — the LLM-side siblings of the same multi-reward imbalance problem.
