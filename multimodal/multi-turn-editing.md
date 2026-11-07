# Multi-turn Image Editing (Drift and TailSteer)
*Depth — the drift problem in iterative image editing, and trajectory-tail steering as the fix.*

**TL;DR:** Multi-turn image editing (edit → edit → edit, chat-style) accumulates drift: subjects warp, identity shifts, colours slide. **TailSteer** (Li et al., 2026) addresses this by fine-tuning *only the trajectory tail* (the last few denoising steps) on the editor's own rollouts, against an anchor-based multi-turn-drift reward. Early denoising remains close to the base model; corrections concentrate where drift actually shows up. Keeps single-turn quality and visibly improves multi-turn identity preservation across FLUX, SenseNova, and Qwen model families.

**Prereqs:** [dpo](../post-training/dpo.md), [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [step-distillation](./step-distillation.md), [vla](./vla.md)

---

## What it is

A **multi-turn edit** is a sequence $(x_0, p_1, x_1, p_2, x_2, \ldots)$ where $x_0$ is the source image, $p_t$ is the edit instruction at turn $t$, and $x_t$ is the editor's output. **Drift** is the accumulation of unwanted changes across turns — identity shift, style slip, background warp — even when each individual edit is reasonable.

Two structural causes:

1. The editor is trained on single-turn pairs; nothing in training penalizes cross-turn degradation.
2. The denoising trajectory moves the image's *entire* latent each turn, not just the region the edit targets. Small changes to the early (near-noise) portion of the trajectory compound.

## How it works (TailSteer)

### Trajectory split

Split the $T$-step denoising trajectory into **early** steps ($t \in [0, T_{\text{split}}]$, near-noise) and **tail** steps ($t \in [T_{\text{split}}, T]$, near-image). $T_{\text{split}} \approx 0.6 T$ reported.

- **Freeze early:** no gradient through early steps. Guarantees the editor still behaves normally on single-turn edits and never diverges in the hard-to-recover high-noise regime.
- **Train tail:** apply RL / SFT on the tail steps only.

### Anchor-based multi-turn-drift reward

For a training rollout at turn $N$:

- Anchor = turn $N-1$ output $x_{N-1}$.
- Reward = similarity-preserving on identity / background (features from a frozen encoder) **minus** a penalty on content drift relative to the anchor, **plus** the standard instruction-following reward for $p_N$.

The anchor-based term is the key: the editor is explicitly rewarded for keeping "everything the prompt didn't ask to change" close to the previous turn.

### Final-stage training only

TailSteer is positioned as a **final-stage** training framework: start from a production editor, keep the early-stage recipe intact, fine-tune only the tail with the drift reward. Low compute footprint.

## Why it matters

- **Multi-turn is becoming the main UX.** ChatGPT image editing, Google / Adobe Firefly chat flows, SenseNova's chat editor — all are multi-turn surfaces. Drift is the dominant failure.
- **Trajectory-local fix is surgical.** Rather than retraining the editor end-to-end or running a full RLHF pass, TailSteer touches only the steps where the problem lives. The approach generalizes to other diffusion-tail problems (artifact removal, text rendering fidelity).
- **Argues for anchor-based multi-turn evals.** The right eval is not "is turn-5 good?" but "is turn-5 close to turn-4 on everything the prompt did not target?"

## Gotchas & tricks

- **$T_{\text{split}}$ is sensitive.** Too early → training affects image composition (regressing on instruction-following); too late → no capacity to correct drift. 0.5–0.7 of $T$ reported.
- **Anchor choice matters.** Previous-turn as anchor is simplest; source image $x_0$ as anchor is stricter but can over-constrain creative edits. Compound anchors (weighted previous + source) are a middle ground.
- **Training rollouts must come from the editor itself.** Rollouts from a stronger editor introduce distribution shift; TailSteer is on-policy by construction.
- **Does not fix identity shift on the first turn.** If turn-1 already changes identity, TailSteer can't recover — it needs an anchor that was correct to begin with.
- **Model-family portability.** Works on FLUX, SenseNova, Qwen (per paper). On older UNet-based editors without flow-matching, $T_{\text{split}}$ may need re-tuning.

## Sources

- Paper: *TailSteer: Trajectory-Tail Steering for Drift-Resistant Multi-turn Image Editing* — Li et al., MMLab-CUHK, 2026 — [arXiv:2611.06741](https://arxiv.org/abs/2611.06741).
- Background: *FLUX* and *SenseNova* editor architectures.
- Background: [step-distillation](./step-distillation.md) — adjacent trajectory-level training recipe in diffusion.
