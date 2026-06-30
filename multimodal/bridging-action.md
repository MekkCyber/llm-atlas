# Bridging-Action Representations for Human→Robot Transfer
*Depth — action-space design for transferring manipulation skills from human video to parallel-gripper robots.*

**TL;DR:** Instead of treating humans as just another 6-DoF bi-manual embodiment (with noisy hand-pose estimates and contact patterns that don't match a parallel gripper), define a **minimal shared action**: the relative wrist translation in the initial head-camera frame. This translation is something humans and robots can both produce cleanly. Train a π₀-like VLA with interleaved action tokens and attention masking so the policy gracefully handles components that one embodiment can't produce.

**Prereqs:** [multimodal README](../multimodal/README.md)
**Related:** [qwen2-5 case study](../case-studies/qwen2-5.md)

---

## What it is

An action-space recipe for VLAs that learn from human data. The goal is not to estimate the human's full pose more accurately — it's to *give up* on the parts that don't transfer (finger contacts, rotations under self-occlusion) and keep only the components the target embodiment can faithfully act on.

## How it works

**The bridging action.** For each timestep, extract the wrist position in the head-mounted camera's frame at `t=0` and compute the relative translation `Δp_t = p_t - p_0`. This signal:

- Requires no rotation estimate (the noisiest hand-pose component from monocular video).
- Is meaningful for both human bi-manual demonstration and a parallel-gripper robot.
- Stays in a single shared coordinate system across embodiments.

**Model.** π₀-style VLA: vision-language backbone with an action head that emits interleaved action tokens. Attention masking allows certain action components to be marked "absent" — important when the human stream has no gripper-state token, or when the robot stream has no finger-joint tokens.

**Training.** Mix human video (translation-only supervision) with robot teleop (full action supervision); the masking handles the mismatch in token availability without architectural surgery.

## Why it matters

- **Scaling with human data.** Bridging-action transfer **scales with the volume of human data**, while noisy 6-DoF transfer does not — a much more favorable data economy for using uncurated human video.
- **Avoids retargeting heuristics.** No need to map human finger contacts to a gripper open/close schedule.
- **Stackable.** Composes with any VLA backbone that supports per-token action masking.

## Gotchas & tricks

- Translation-only loses information about *how* the object is manipulated (orientation matters for many tasks). The paper picks a task suite (parallel-gripper bi-manual) where this trade-off is favorable; tasks requiring dexterous re-orientation will need additional bridging signals.
- Initial-frame anchoring requires reliable head-camera calibration during the first frame; drift or mis-calibration directly corrupts the action trace.
- The attention-masking trick is general and worth borrowing even when not using the wrist-translation signal — it cleanly handles heterogeneous action vocabularies across embodiments.

## Sources

- Paper: *Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots* — arXiv:2606.28133 — https://arxiv.org/abs/2606.28133
