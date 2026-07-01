# Diffusion-MoE Routing (Saliency-Guided)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Diffusion MoEs make routing decisions based on **noisy** latents — but noise obscures the very structural / textural features that would let the router send more compute to salient tokens. **SharpMoE** decouples the routing input from the model input: the noisy latent still flows through the experts, but the router sees a **clean** side-channel latent as its saliency signal. Adds a **trajectory routing loss** to enforce consistent compute allocation across the whole denoising rollout. Plug-and-play post-training on pretrained diffusion MoEs.

**Prereqs:** [_moe.md](./_moe.md), [load-balancing-loss.md](./load-balancing-loss.md)
**Related:** [../multimodal/README.md](../multimodal/README.md), [../multimodal/masked-discrete-diffusion.md](../multimodal/masked-discrete-diffusion.md)

---

## What it is

MoE routers in text LMs see one input distribution (semantic tokens) and route on features they've been trained to notice. Diffusion MoEs are different: at high-noise timesteps the router's input is *mostly noise*, and the salient structure the router *should* route on (edges, objects, text-in-image) is buried. The router can't recover it, so compute is misallocated to non-salient tokens.

SharpMoE's fix: give the router a **clean-latent guidance signal** — a noise-free feature map derived from the same input — as its saliency reference, while the noisy latent stays on the compute path. The router now routes on structure, not on noise.

## How it works

**Two decoupled signals.**
- **Compute path.** Noisy latent → experts (unchanged from a standard diffusion MoE).
- **Routing path.** Clean latent → router. The clean latent is derived from the input at a lower noise level (or from a lightweight auxiliary encoder) — it's fed to the router only, not to the experts.

**Router.** Standard top-k gating over the clean-latent features. Because it sees structure rather than noise, salient tokens (object boundaries, textured regions) receive high routing scores even when the corresponding noisy latent is featureless.

**Trajectory routing loss.**
- A standard diffusion MoE routes each timestep independently; the same token can end up with wildly different compute allocation at t=0.9 vs t=0.1.
- Trajectory routing loss constrains the *shape* of compute allocation across timesteps: salient tokens should receive high compute consistently across the denoising rollout, not just at low-noise steps.
- Instantiated as a smoothness / consistency penalty over per-token routing scores along `t`.

**Plug-and-play.** Applied post-training to a converged diffusion MoE — no from-scratch retraining. Trains only the router (and possibly a small clean-latent extractor) while the experts are frozen.

**Reported result.** SOTA visual generation quality as a plug-and-play patch to converged MoE image models.

## Why it matters

- **Fixes a specific structural mismatch.** MoE routing gets easier as inputs become more informative — pointing the router at cleaner features is a general lever, not specific to diffusion.
- **Trajectory consistency is a new class of routing loss.** The denoising axis is unique to diffusion; the "compute allocation should be a coherent function of `t`" principle plausibly generalises to other iterative decoders (semi-AR text generation, agent multi-step reasoning).
- **Retrofit path for existing diffusion MoEs.** Labs with expensive-to-retrain converged MoE image models get a router-only upgrade.

## Gotchas & tricks

- **Clean-latent source matters.** Using the ground-truth clean latent at training time is fine; at inference the clean latent isn't available, so a proxy (auxiliary encoder or lower-noise sample) is needed — quality depends on the proxy fidelity.
- **Load-balancing still applies.** The saliency signal biases routing toward salient tokens; standard load-balancing losses ([load-balancing-loss.md](./load-balancing-loss.md)) are needed on top to prevent expert collapse.
- **Trajectory loss coefficient is load-bearing.** Too weak → per-step routing drifts; too strong → the router becomes near-static across timesteps and misses per-timestep specialisation.
- **Not verified for text or video diffusion.** The paper's evaluation is image generation; transferability to text-diffusion or video-diffusion MoEs is an open question.

## Sources

- Paper: *Focusing on What Matters: Saliency-Harnessing Accurate Routing for Diffusion MoE* — Yan, Mao, Wang, Liu, Gao, Sang (HUST / Tongyi Lab, Alibaba), 2026 — [arXiv:2606.26938](https://arxiv.org/abs/2606.26938).
