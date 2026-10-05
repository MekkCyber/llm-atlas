# HC-DLM — Hierarchical Continuous Diffusion Language Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A hierarchical diffusion language model where a **continuous latent trajectory** is the only persistent generative state. Discrete tokens are read out from the latent at every denoising step and fed back as a scaffold for the next latent update. The denoiser never relies on independent per-position marginals (the usual discrete-diffusion failure mode) and never loses grounding in a valid token configuration (the usual continuous-diffusion failure mode).

**Prereqs:** [transformer-block.md](./transformer-block.md)
**Related:** [e-moe.md](./e-moe.md), [_moe.md](./_moe.md)

---

## What it is

Ren et al. (UIUC, 2026) address a structural limitation of parallel-decoding diffusion LMs. Discrete diffusion LMs sample each token from its marginal — breaking statistical dependencies among tokens decoded together. Continuous diffusion LMs avoid that by denoising a shared continuous state, but the denoiser sees only the state, so nothing ties it to a valid token configuration until the final decode. HC-DLM couples the two in one principled denoising process.

## How it works

- **Single persistent state.** The latent $z_t$ is the only state carried across denoising steps. Tokens $x_t$ are a readout from $z_t$, not an independent chain.
- **Hierarchical denoising.** At each step: (1) decode current tokens from $z_t$, (2) use those tokens as a scaffold for the next latent update $z_{t+1}$, (3) re-decode. Tokens feed back into the latent, closing the loop.
- **Training objective.** Derived from a variational bound on the token log-likelihood — the authors show the hierarchical readout fits inside the ELBO without a separate reconstruction term.
- **Contrast with prior work.** Methods that attach continuous context to a self-contained discrete chain treat the latent as auxiliary; HC-DLM makes the latent primary and the tokens the readout.

## Why it matters

- Beats both discrete and continuous diffusion LM baselines at matched model size on three benchmarks: higher Sudoku and Countdown puzzle accuracy (structured reasoning / planning), lower generative perplexity on LM1B.
- Few-step decoding is the regime where diffusion LMs can beat AR on latency; HC-DLM's hierarchical coupling directly attacks the inter-token coherence problem that has blocked them from winning on quality.
- Pairs naturally with MoE-as-latent variants (see [e-moe.md](./e-moe.md)) — orthogonal ways of injecting structure into the few-step reverse process.

## Gotchas & tricks

- Readout at every step adds compute vs standard continuous diffusion; the win vs AR requires the step count to be small enough that the hierarchical loop still beats autoregressive decoding.
- Variational bound requires the readout model to be tractable; the paper uses a lightweight head.
- Results so far are on structured-reasoning puzzles (Sudoku, Countdown) and LM1B — a smaller-scale LM benchmark. Scaling to full LM pretraining is open.
- Project page: [https://hc-dlm.github.io/](https://hc-dlm.github.io/).

## Sources

- Paper: *Hierarchical Continuous Diffusion Language Models* — Ren, Li, Liu, Liu, Schwing; UIUC, 2026 — [arXiv:2610.02193](https://arxiv.org/abs/2610.02193).
- Context: *Masked Diffusion Language Models* (MDM line, Lou et al. 2023 et seq.) — the discrete-diffusion ancestor whose per-position marginal factorisation this paper abandons.
- Context: continuous diffusion LMs — Li et al. 2022, Dieleman et al. 2022 — the continuous ancestors whose "latent sees only itself" problem this paper addresses.
