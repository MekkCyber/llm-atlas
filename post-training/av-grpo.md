# AV-GRPO — Modality-Anchored Decoupled GRPO
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO variant for **joint audio-video generation**: instead of a single joint reward that mixes cross-modal and within-modal quality (which makes RL unstable), AV-GRPO factors the preference signal into modality-anchored sub-rewards — audio fidelity, video fidelity, and cross-modal synchronisation — and runs GRPO with modality-aware advantages while sharing the diffusion policy.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [_rewards.md](_rewards.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Joint audio-video generative models (LTX-2.3 class) suffer from three coupled failures: per-modality fidelity is limited, text-alignment is weak, and audio-video sync is fragile. A single-scalar preference reward tries to fix all three at once and typically **oscillates** — RL push-pulls between improving audio, improving video, and improving sync, none of them converging.

AV-GRPO decouples this by decomposing the multimodal preference into three independent sub-preferences:

- $r_A$ — anchored on audio quality only (video held fixed).
- $r_V$ — anchored on video quality only (audio held fixed).
- $r_{AV}$ — cross-modal sync (both attended).

Each sub-preference is a separate GRPO-style reward stream. The policy is shared, but advantages are computed per stream and summed:

$$A_i = w_A A_i^A + w_V A_i^V + w_{AV} A_i^{AV}$$

with $A_i^k = (r_i^k - \bar r^k) / \sigma^k$ inside each modality's group.

## How it works

The paper uses a **5DAV** dataset: pairs of generated audio-video clips with three preference labels (audio-only, video-only, sync). At each RL step:

1. Sample $G$ audio-video generations per prompt from the joint diffusion policy.
2. Score each with the three preference models trained on 5DAV.
3. Compute per-stream group-relative advantages independently.
4. Combine into $A_i$ and apply the standard GRPO clipped-ratio update with a KL penalty to the pre-RL checkpoint.

Because the policy is a diffusion model, the clipped ratio is on the *per-step denoising distributions*, following the diffusion-GRPO recipe (importance ratio at each denoising step, PPO clip applied per step, advantage broadcast).

## Why it matters

- **Stabilises multimodal RL.** Ablations in the paper show that with a single joint reward, the RL run diverges after a few epochs; decoupling into modality-anchored sub-rewards recovers stable training.
- **Improves each axis.** AV-GRPO beats LTX-2.3 baseline on all three axes — per-modality FVD/audio fidelity, text alignment, and audio-video sync — rather than trading one against another.
- **Recipe generalises.** The same decoupling pattern applies to any multimodal generator whose preferences can be attributed to a subset of modalities.

## Gotchas & tricks

- **Sub-reward independence is a strong assumption.** In practice, audio-only and video-only scores are correlated; the paper reports mild collinearity and adds a whitening step across sub-advantages.
- **Weights matter more than in text GRPO.** $w_A, w_V, w_{AV}$ are tuned on a held-out preference set; a bad ratio collapses AV-GRPO to single-modality optimisation.
- **Preference models are per-modality models.** Training three preference models is expensive; AV-GRPO amortises this by re-using a shared audio-video encoder trunk with three task-specific heads.
- **KL to reference is essential.** Without it, the diffusion policy loses text alignment even as fidelity improves — reward hacking on non-textual axes.

## Sources

- Paper: *AV-GRPO: Modality-Anchored Decoupling Diffusion Reinforcement Learning for Joint Audio-Video Generation* — Xu, Yan, Shi, et al., 2026, [arXiv 2609.29816](https://arxiv.org/abs/2609.29816) — introduces AV-GRPO and the 5DAV preference set.
- Paper: *DeepSeekMath* — Shao et al., 2024 — the GRPO baseline.
