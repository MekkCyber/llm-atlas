# TailSteer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Iterative image editing — repeatedly asking an editor to apply edits to its own outputs — drifts over 15–20 turns: colors shift, identities deform, structure corrodes, even when each single edit looks reasonable. TailSteer is a final-stage training recipe that **keeps early denoising dynamics close to the base model** and concentrates the editor's corrections near the *tail* of the diffusion trajectory. Training data is self-generated via the editor's own multi-turn rollouts; no curated editing pairs. Introduced in 2611.06741.

**Prereqs:** none
**Related:** none

---

## What it is

A fine-tuning strategy for diffusion image editors that regularizes the fine-tuned score function to match the base model on **early** diffusion timesteps and only diverge on **late** timesteps. The reasoning: early-step scores set the global structure of the sample; late-step scores add detail. Drift accumulates across turns because fine-tuned editors shift the *whole* trajectory, slowly biasing the global distribution. Freezing the early trajectory and training only the tail prevents this.

## How it works

For a diffusion model $s_\theta(x_t, t)$ fine-tuned from a base $s_{\theta_0}$:

1. **Collect rollouts.** Run the current editor through many multi-turn editing sequences on the editor's own outputs — no curated reference edits needed.
2. **Define a timestep split** $t^* \in (0, T)$ (the "tail boundary").
3. **Loss.**
   - For $t < t^*$ (late denoising, close to clean image): a standard editing loss that moves the score toward correct edits.
   - For $t \ge t^*$ (early denoising, close to pure noise): an anchor loss $\|s_\theta(x_t, t) - s_{\theta_0}(x_t, t)\|^2$ that pins the fine-tuned score to the base.

$$\mathcal{L} = \mathbb{E}_{t, x_t}\Big[\mathbb{1}[t < t^*] \cdot \mathcal{L}_{\text{edit}}(s_\theta, x_t, t) + \mathbb{1}[t \ge t^*] \cdot \lambda \cdot \|s_\theta - s_{\theta_0}\|^2\Big]$$

The editor rolls out its own multi-turn sequences to produce the training distribution; this makes the recipe self-supervised and avoids the need for drift-free reference trajectories.

## Why it matters

- **Clean statement of a well-known failure.** Iterative edit drift was a known pain in production image-editing systems; TailSteer gives it a mechanical explanation and a mechanical fix.
- **Base-fidelity preserved.** Because the early trajectory is anchored, the fine-tuned editor retains the base model's distribution for cases where no edit is needed.
- **Generalizable split.** The "freeze head, train tail" decomposition should apply to RLHF on diffusion (preference-tune the tail only), consistency distillation (distill the tail), and multi-stage sampling pipelines in general.

Reported: substantial drift reduction over 15–20-turn edit sequences on FLUX, SenseNova, and Qwen model families, with base-model performance largely preserved.

## Gotchas & tricks

- The tail-boundary $t^*$ is a hyperparameter — too low and drift returns, too high and the editor can't change enough to apply an edit. Sweep per model family.
- Self-generated rollouts can amplify biases of the base editor; monitor drift during fine-tuning by periodically evaluating a held-out multi-turn benchmark.
- Anchor loss weight $\lambda$ competes with the editing loss; too large and the editor becomes a no-op.
- Compatible with LoRA-style adapters — only the tail-timestep parameters need updating.

## Sources

- Paper: TailSteer: Trajectory-Tail Steering for Drift-Resistant Multi-turn Image Editing — Li et al., 2026 — [arXiv:2611.06741](https://arxiv.org/abs/2611.06741)
