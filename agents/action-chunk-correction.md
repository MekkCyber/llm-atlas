# Action-Chunk Correction (VLA-Corrector)
*Depth — a lightweight monitor + online gradient guidance for closed-loop VLA action chunks.*

**TL;DR:** VLA models plan actions in chunks — typically 8 or 16 actions predicted at once, then executed. Long chunks amortize planning cost but destroy reactivity: if the world diverges mid-chunk, the plan is stale. VLA-Corrector adds a small **latent-space vision monitor** trained to detect this divergence, and triggers **Online Gradient Guidance** — a few gradient steps on the remaining actions at inference time — to correct without discarding the chunk. Adaptive action horizon = the effective chunk length shrinks under monitor uncertainty. Improves robustness on contact-rich manipulation with minimal added latency.

**Prereqs:** [_vla](_vla.md)
**Related:** [../inference/README.md](../inference/README.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Chunked action decoding is the standard efficiency trick in VLA policies: instead of predicting one action per environment step, predict $k$ actions ($k = 8, 16$, sometimes more) and execute them open-loop before the next inference pass. This amortizes the transformer forward pass over $k$ environment steps and hits control frequencies open-loop can never reach.

The failure mode is direct: physical execution can diverge from the model's implicit prediction. Contact events, slippage, sensor noise — the state after action 3 is not what the model assumed when it planned action 4. Playing the remaining $k - 3$ actions blindly makes the policy brittle in exactly the settings (contact-rich manipulation) where robots need robustness.

Naïve fix: shrink the chunk. But small chunks lose most of the efficiency gain. VLA-Corrector's fix: keep the chunk large, but *monitor* execution and *correct* when needed.

## How it works

**Monitor head.** A small classifier trained on the VLA's own latent-space features to score, at each executed action step, whether the current observation is consistent with the model's internal expectation for that step. Positive samples: on-policy executions the base VLA plans successfully. Negative samples: perturbed or off-distribution executions. Output: a divergence score per step.

**Trigger condition.** If the monitor's divergence score crosses a threshold on step $t$ of a chunk, action-chunk correction fires. Below threshold: continue executing the chunk open-loop.

**Online Gradient Guidance (OGG).** When triggered, run a handful of gradient steps on the *remaining* actions in the chunk at inference time. The loss: a differentiable score that expresses "current observation matches the expected observation for the corrected action." Update only the remaining actions (not the model weights) — this is a few-step test-time optimization, like classifier guidance in diffusion. After OGG, continue executing the corrected tail.

**Adaptive action horizon.** The effective chunk length becomes state-dependent. Confident regions (open air, free motion) execute the full chunk; uncertain regions (fingertip in contact with a surface) trigger frequent OGG and effectively shorten the chunk. The nominal chunk stays large; the effective horizon adapts.

## Why it matters

- **Preserves chunk-decoding efficiency without giving up reactivity.** A pure short-chunk baseline runs the transformer too often; a pure long-chunk baseline is brittle. VLA-Corrector picks the good parts of both.
- **Model-agnostic plug-in.** Works on any VLA that emits action chunks — no retraining of the base policy. The monitor is a small head; OGG operates through the base VLA's forward+backward pass.
- **Generalizes beyond robotics.** The same monitor + test-time gradient-correction pattern applies to any speculative-multi-step decoding — speculative decoding for LLMs, action-chunked agents, planning-based agent trajectories.
- **Cheap.** Monitor scoring is one small forward per step; OGG runs only when the monitor triggers, so its cost is amortized over the "good" steps.

## Gotchas & tricks

- **Threshold calibration.** Too strict → OGG fires constantly → back to per-step latency. Too loose → misses genuine divergence. The paper does not specify a universal threshold; expect per-task calibration.
- **Monitor false negatives are worse than false positives.** A false-positive OGG costs a few extra ms; a false-negative continues executing a bad plan through a contact event. Bias the threshold toward triggering.
- **OGG requires backprop through the action head.** Some frozen serving stacks disable gradients; enabling them for OGG adds ~2× memory. In practice, only the head + last few layers need gradients.
- **Interaction with quantized deployment.** OGG's gradient step is sensitive to numerical precision; INT4/FP8 deployment can degrade correction quality. Consider mixed precision only for the OGG path.
- **The monitor's training data.** Positive samples are easy (on-policy VLA rollouts); negative samples need synthesis — perturbed executions, adversarial rollouts, or replay-buffer trajectories from failed episodes. Poor negative sampling → weak monitor.

## Sources

- Paper: *VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon* — Pan et al., ZJU-OmniAI, 2026 — [arXiv:2607.01804](https://arxiv.org/abs/2607.01804)
- Related: π0 / π0.5 (flow-matching action heads), OpenVLA (open baseline), Diffusion Policy (chunked action prediction with diffusion).
