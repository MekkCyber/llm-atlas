# Video World Models
*Depth — video generators that model how the world evolves, not just what it looks like.*

**TL;DR:** Video diffusion models fit pixel distributions but chronically fail to *extrapolate* physics — hand them an out-of-distribution initial condition and the "world" collapses into flicker. Video-world-model architectures treat the latent trajectory as a dynamical system: a physics-motivated prior handles the low-order motion, and the network only has to learn the residual. LDR (2026) is a clean instance and reports **>20×** smaller in- vs. out-of-distribution error gaps than baseline video diffusion at a fraction of the parameters.

**Prereqs:** [../multimodal/README.md](README.md)
**Related:** [../evaluation/long-horizon-consistency.md](../evaluation/long-horizon-consistency.md)

---

## What it is

A video generator whose latent transition function is designed around a *dynamics prior* — position-plus-velocity integration in latent space — with a learned residual for higher-order effects. The point is not to render photorealistic frames but to make future frames obey the same laws that governed past frames. That property is what "world model" is usually shorthand for.

## How it works

For a latent sequence $z_1, z_2, \dots, z_t$, the naive video-diffusion approach learns $p(z_{t+1} \mid z_{\le t})$ directly. LDR-style architectures split the transition:

1. **Kinematic prior.** Compute a first-order (position + velocity) integration step in latent space: $\hat z_{t+1} = z_t + \Delta t \cdot v_t$, with $v_t$ estimated from the immediate past.
2. **Learned residual.** The network regresses only $z_{t+1} - \hat z_{t+1}$ — higher-order dynamics the linear prior can't capture (acceleration, contact forces, discrete state changes).
3. **Diffusion or flow-matching over the residual.** Standard generative modeling machinery on the (much lower-variance) residual distribution.

Because the prior handles the low-order structure exactly, the residual is smaller, easier to learn, and generalizes further out-of-distribution.

## Why it matters

- **The "video model = world model" claim keeps breaking on physics.** Explicit dynamics priors are the plausible route to video generators that predict rather than just interpolate.
- **Parameter efficiency.** LDR reports fewer parameters and faster inference than video-diffusion baselines while matching or exceeding OOD accuracy.
- **Composability with agents.** A predictive video model is a natural forward-simulator for planning-style agents; the physics gap is what has kept that combination from working.
- Adjacent to the world-model research agenda benchmarked by AutoWorldModel-Bench (2026) — that benchmark uses structured-state worlds, this line targets pixel-space worlds with structured-latent priors.

## Gotchas & tricks

- **The prior is only as good as its integration frame.** Fixed $\Delta t$ breaks for videos with variable inter-frame motion; scale the prior by an estimated inter-frame $\Delta t$.
- **Residual regression collapses.** If the network is over-parameterized relative to the residual scale, it will learn to *undo* the prior. Regularize the residual head.
- **In-distribution wins are muted.** The advantage shows up in extrapolation — evaluating only on IID frames will make this look like a lateral move.
- **Latent-space physics is not pixel-space physics.** The kinematic prior is meaningful only when the latent encoder preserves geometric structure; VAEs with strong perceptual losses can wash it out.

## Sources

- Paper: *Learning How the World Evolves: Extrapolative Video World Models via Latent Dynamics Reasoning* — Li, Liu, Wang, Ge, Ji, Zhang, Lin, Lu, Lin, Chandraker, 2026 — [arXiv:2608.09926](https://arxiv.org/abs/2608.09926) — LDR (Latent Dynamics Reasoning) formulation and the physics-benchmark evaluation.
