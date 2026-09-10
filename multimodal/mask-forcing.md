# Mask Forcing
*Depth — a dual-noise masking rollout strategy that widens teacher-distribution coverage in autoregressive video-diffusion distillation.*

**TL;DR:** Distribution Matching Distillation (DMD) applied to AR video diffusion collapses onto a few teacher modes because reverse-KL is mode-seeking, giving over-saturated and over-smoothed videos. Mask Forcing perturbs the student's self-rollout by randomly masking tokens along both spatial and temporal axes and re-noising them from cleaner draws, so the student explores under-covered teacher regions and the cleaner tokens double as denoising guidance for their noisier neighbours. No real video data, no extra post-training stage.

**Prereqs:** [attention.md](../fundamentals/attention.md)
**Related:** *(no other diffusion depth files in the graph yet)*

---

## What it is

A drop-in modification of the self-rollout step used in AR-video diffusion distillation (DMD family). The student is trained to match a bidirectional teacher's distribution via reverse KL, and mode collapse is the standard failure mode. Mask Forcing changes what the student sees during rollout — not the loss.

## How it works

Standard AR-video DMD self-rollout: the student autoregressively predicts the next chunk of noisy video tokens, and the reverse-KL gradient is computed at each step.

Mask Forcing modifies each rollout step:

1. **Select a random mask** over the noisy input tokens along **both** spatial and temporal axes. Mask fraction and per-axis mixing are hyperparameters.
2. **Replace masked tokens** with samples drawn from a cleaner (lower-noise) noise level than the current rollout step — the "dual-noise" in the name.
3. Feed the mixed clean/noisy input into the student and compute the standard DMD loss against the teacher.

Effect: (a) the student rollout no longer collapses onto its current dominant modes because a fresh clean signal steers it into other basins of the teacher; (b) intermediate denoising is easier because clean tokens locally guide the noisy ones, reducing autoregressive error accumulation over long horizons.

## Why it matters

- **Attacks the standard AR-video DMD failure mode.** Reverse-KL mode collapse is the reason distilled AR video looks over-smoothed and over-saturated relative to the bidirectional teacher.
- **Data-free.** No real video, no additional post-training pass — the trick is entirely inside the existing distillation loop.
- **Composable.** Works on top of existing DMD variants; not tied to a specific student or teacher architecture.
- **Transferable in principle.** Reverse-KL mode collapse is the same problem in image diffusion distillation and in language-model distillation — the mask-forcing recipe is a candidate for those settings too.

## Gotchas & tricks

- **Mask ratio is a knob.** Too low and coverage doesn't widen; too high and the student learns to rely on unrealistic clean-token supervision. Sweep 20–50%.
- **Mask along both axes.** Spatial-only masks leave temporal error accumulation intact; temporal-only masks miss regional mode collapse. Random on both is the setup that wins.
- **Cleaner noise level is a schedule.** Use noise levels below the current rollout step but not zero — full-clean substitutions leak information the student wouldn't have at inference.
- **Doesn't fix teacher failures.** If the bidirectional teacher itself is biased, wider coverage of the teacher just amplifies that bias.

## Sources

- Zhao et al., *Mask Forcing: Improving Autoregressive Video Diffusion Distillation via Dual-Noise Masking Rollout* — 2026 — https://arxiv.org/abs/2609.09123
- Yin et al., *One-step Diffusion with Distribution Matching Distillation (DMD)* — 2023 — https://arxiv.org/abs/2311.18828
