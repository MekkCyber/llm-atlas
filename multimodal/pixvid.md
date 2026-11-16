# PixVid
*Depth — lifting a latent video diffusion model into pixel space by swapping the decoder.*

**TL;DR:** Latent video diffusion runs the DiT in a VAE's compressed latent space, which is efficient but inherits that VAE's reconstruction ceiling. PixVid keeps the pretrained DiT (Wan2.1-14B) and replaces the VAE decoder with a convolutional head that renders directly into pixels. Two training-side ingredients matter: predict the *clean* video target (not velocity) and increase the noise shift. Generates at twice the resolution with the same per-frame token count as the latent baseline.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [g-star.md](g-star.md)

---

## What it is

A finetuning recipe that turns a latent video DiT into a pixel-space video generator. Only the pixel-rendering head is retrained; the DiT weights (3D attention, temporal conv, text conditioning) are reused. The move removes a fixed artifact: whatever the VAE decoder cannot represent, the latent model cannot produce — however good the DiT is.

## How it works

- **Decoder swap.** The frozen DiT's final latent output is fed into a new convolutional head that upsamples and renders RGB pixels directly. The VAE decoder is discarded at inference.
- **Clean-target prediction.** Standard latent DiTs are trained to predict velocity (v-prediction). PixVid switches to predicting the clean video and reports this is key for pixel-space quality — velocity prediction mis-balances signal vs noise when the output distribution is pixels rather than latents.
- **Larger noise shift.** The noise schedule's shift parameter is increased so more training steps land in the high-noise regime where coarse structure is set. This matters more in pixel space, where fine detail needs a clean coarse scaffold.

Finetuning is initialized from Wan2.1-14B; the rest of the stack is left alone.

## Why it matters

Practical path off the latent-diffusion ceiling for video. Also a proof that the latent → pixel migration doesn't require retraining the generator — just the output head and the training objective. For deployment, the per-frame token count is unchanged, so attention compute is unchanged; the extra cost is in the pixel-rendering head, which is comparatively cheap.

## Gotchas & tricks

- "Clean target" matters more than it sounds — the switch is not cosmetic. Velocity prediction underweights the high-frequency part of the loss in pixel space.
- Noise-shift tuning is coupled to resolution. Raising resolution without re-shifting pushes the model into a regime it hasn't trained on.
- The pixel head's architecture (receptive field, upsampling) is where most of the remaining engineering lives. The paper has specific choices; naive conv heads leak VAE-like artifacts.
- Temporal consistency does not come for free. The DiT handles it in latent space; the pixel head can still introduce flicker if its temporal receptive field is too shallow.
- The recipe is a *finetune*, not a train-from-scratch — attempting to apply the clean-target objective to a cold-start DiT does not give the same gains.

## Sources

- Paper: *Back to Pixels: Lifting Latent Video Diffusion into Pixel Space* — Song, Chen, Rockwell, He, Liu, Xie, Gu, Park, V S, Menn, Yang, Huang (Roblox Corporation), 2026 — [arXiv:2611.17336](https://arxiv.org/abs/2611.17336)
