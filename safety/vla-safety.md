# VLA Safety and Safe Fallback
*Depth — making a vision-language-action policy refuse the *behavior*, not just the initial prompt.*

**TL;DR:** A VLA model asked to perform a prohibited task will typically refuse at step 0 — but if an external controller nudges the robot into a mid-execution state of that prohibited task, the policy often completes it. **State-Aware Fallback Imitation (SAFI)** trains a policy whose target action for prohibited-task states is a *safe hold* (no end-effector motion, no object release), jointly with the original actions for permitted tasks. On LIBERO tasks this cuts post-takeover prohibited-task success to **1.3%** with no recorded physical incidents.

**Prereqs:** [refusal-suppression](refusal-suppression.md)
**Related:** [_attacks](_attacks.md) · [cot-monitoring](cot-monitoring.md)

---

## What it is

Two observations about current VLA refusals:

1. **Prompt-level refusal is not state-level refusal.** Current VLAs refuse the prohibited task when evaluated from step 0. They do not refuse when evaluated from an intermediate state of the same task.
2. **Control takeover is a realistic adversary model.** An external operator (or another process) can set the robot's state to mid-execution of a prohibited behavior; the VLA then finishes it under its normal control loop.

SAFI formalizes the refusal as a **state-conditioned safe-hold behavior**: for prohibited-task-adjacent states, the policy's target action is to not move — specifically no end-effector displacement and no change in grasp state — regardless of the prompt.

## How it works

Training data augmentation:

- Collect trajectories of the prohibited tasks and label states along them.
- For each such state, replace the next-action target with a **safe hold**: zero end-effector velocity, retain grasp (don't drop a held object).
- Train jointly with the original permitted-task demonstrations under the standard VLA imitation objective.

Behaviorally, this trains the policy to treat "being in a prohibited-task state" as a trigger for a safe halt, independent of what the current prompt says to do. The fallback is thus **state-triggered, not prompt-triggered**.

A subtle design point: the "hold" must be safe, which means:

- No end-effector motion (don't jerk the arm).
- No release (a held object stays held — releasing could drop it).
- No actuator saturation (don't drive to joint limits trying to zero out drift).

## Why it matters

- **Breaks the control-takeover attack.** This attack is the embodied analogue of "deceptive compliance": refusing the request without refusing the behavior. SAFI treats refusal as behavior-level, not surface-level.
- **Minimal retraining.** Needs trajectories of prohibited tasks plus the augmentation; no new model class, no RL, no reward model.
- **Composable with prompt-level refusal.** SAFI is additive; a VLA can refuse at the prompt *and* hold under takeover.
- **An existence proof that embodied refusal is learnable.** Changes the baseline for VLA safety from "prompt-level mitigations only" to "behavior-level refusal is cheap enough to deploy."

## Gotchas & tricks

- **"Prohibited" must be labeled.** Needs a state classifier or annotated trajectories defining what counts as mid-execution of a prohibited task. Scope-limited by that labeling.
- **Fallback behavior is task-dependent.** On LIBERO, "hold" is simple. For mobile manipulation or locomotion, "hold" may itself be unsafe (a moving robot that stops abruptly can fall). The fallback must match the morphology.
- **Doesn't prevent all harm vectors.** An attacker who can set the state can usually also override the control output. SAFI is defense against passive takeover, not active override of the actuation signal.
- **Covered-task leakage.** If a permitted task shares intermediate states with a prohibited task, SAFI can over-trigger on permitted tasks. The paper manages this via careful state-labeling.

## Sources

- Paper: *Teaching Vision-Language-Action Models to Stop Safely* (SAFI) — Wang, Guo, Huang, Yuan, Ma, Jiang, 2026 (Notre Dame · Georgia Tech · MIT).
