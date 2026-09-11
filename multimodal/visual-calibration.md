# Visual Calibration for World Models

*Depth — teach a world model to interpret actions through a short in-context "calibration episode" so it generalizes zero-shot to unseen environments and embodiments.*

**TL;DR:** The same numerical robot action produces different pixels depending on camera view, robot placement, and embodiment — so a world model trained across mixed setups gets brittle. Visual calibration solves it in-context: a short **calibration episode** (a demonstration clip showing how actions look in *this* setup) is prepended to the world-model context. Conditioned on that clip, the model interprets subsequent actions correctly in the target setup. Enables zero-shot simulation on unseen scenes/embodiments and test-time policy improvement without any weight updates.

**Prereqs:** [programmable-world-model](programmable-world-model.md), [../post-training/wmrl.md](../post-training/wmrl.md)
**Related:** [README](README.md)

---

## What it is

Actions in embodied domains are not a universal language in pixel space. A "move joint 3 by +0.05 rad" instruction on a Franka arm looks completely different from the same instruction on a Kuka arm, and even the *same arm* under a different camera looks different. A single world model trained on a mixed corpus of embodiments/views receives conflicting supervision — the same numeric action maps to many possible visual continuations, and the model learns some blur of them.

Visual calibration moves the action-to-visual mapping from *weights* to *context*: at test time, hand the world model a short clip showing how actions look in the target setup, and let it condition on that clip.

## How it works

Training:

- Corpus contains multi-embodiment / multi-view episodes.
- For each training window, a short prefix ("calibration clip") is drawn from the same setup as the target rollout.
- The world model is trained to predict the target rollout *conditioned on* the calibration clip + past frames + action sequence.
- With calibration explicitly present in training, the model learns to interpret actions *through* the calibration clip rather than by averaging over training setups.

Inference in an unseen setup:

```
calibration = record_short_clip(new_setup)   # e.g. a few random actions
state_0     = init(new_setup)
for t:
    obs_t   = world_model(calibration, past_frames, action_t)
    action_t+1 = policy(obs_t)
```

If a calibration clip is unavailable, an *interaction-history fallback* re-uses the initial frames of the current episode as an implicit calibration signal.

## Why it matters

- **Zero-shot cross-embodiment simulation.** Same weights, new arm/room/camera — the model still simulates correctly given a calibration clip.
- **Test-time policy improvement.** Because the calibration is in-context, a policy can be improved against the simulator with *no* re-training of the world model.
- **Composes with WMRL and PWM.** [WMRL](../post-training/wmrl.md) needs a world model that generalizes; [programmable-world-model](programmable-world-model.md) needs the visual renderer to interpret actions correctly. Visual calibration is the missing link.

## Gotchas & tricks

- **Calibration clip must reveal the mapping.** A clip that only shows static scenes is uninformative — the calibration set needs varied actions.
- **Length tradeoff.** Longer calibration = better conditioning but tighter context budget. The paper reports short clips (seconds) are sufficient.
- **Interaction-history fallback drifts.** If the first observations don't span the action space, later frames will be noisy — an actively-collected calibration episode is safer.
- **Doesn't help with unseen *dynamics*.** Calibration teaches "what does this action look like"; it doesn't teach new physics. Radically new embodiments (e.g. soft-body) still need retraining.

## Sources

- Paper: *SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators* — Yang, Han, Ozyurt, Yang, Yang, Cao, Zhen, Du, Gan — UC Berkeley / NYU / UMass / Harvard, 2026 — [arXiv:2609.09155](https://arxiv.org/abs/2609.09155).
