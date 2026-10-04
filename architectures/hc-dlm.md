# Hierarchical Continuous Diffusion Language Models (HC-DLM)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Discrete diffusion LMs sample each unmasked token from its marginal, breaking token-token dependencies when decoding in parallel. Continuous diffusion LMs avoid that but their denoiser only sees the latent state and never a valid token configuration until the final step. **HC-DLM** (Ren et al., 2026) couples a persistent continuous latent trajectory with a discrete token read-out at *every* step; tokens are emitted from the latent and fed back as a scaffold for the next latent update. Trained with a variational bound on token likelihood; beats discrete and continuous baselines at matched size on Sudoku, Countdown, and LM1B.

**Prereqs:** [transformer-block.md](./transformer-block.md), [multi-head-attention.md](./multi-head-attention.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Two existing families of diffusion LMs:

| Family | Primary state | Weakness |
| --- | --- | --- |
| Discrete (MDLM, LLaDA, SEDD) | Token sequence masked over time | Parallel decode samples tokens independently from marginals |
| Continuous (Plaid, Diffusion-LM) | A continuous latent vector | Denoiser never conditioned on a valid token config |

HC-DLM unifies them: *one* denoising process with a continuous latent as the persistent generative state, and discrete tokens read off at every step and fed back as a scaffold. Not a chain of continuous context attached to a self-contained discrete chain (as in recent hybrids) — the latent is **the** state.

## How it works

Denote the continuous latent at diffusion time $t$ as $z_t$ and the readout tokens as $x_t$. One denoising step:

1. **Read.** Decode tokens $x_t$ from $z_t$ using a token-head.
2. **Scaffold.** Feed $x_t$ back as structured conditioning for the next latent update.
3. **Update.** Produce $z_{t-1}$ by running the denoiser on $(z_t, x_t)$.
4. **Repeat.** Standard diffusion schedule down to $t = 0$.

The training loss is a variational bound on the marginal token likelihood $\log p(x_0)$:

$$
\mathcal{L} = \mathbb{E}_q[\text{KL}(q(z_{t-1}\!\mid\!z_t, x_0) \,\|\, p_\theta(z_{t-1}\!\mid\!z_t, x_t))] - \mathbb{E}_q[\log p_\theta(x_0 \mid z_0)]
$$

so the latent carries token information all the way through the diffusion, and the read-out tokens are not an afterthought.

## Why it matters

- **Unifies two diffusion-LM families** under a single principled objective (variational bound on tokens, with a persistent latent).
- **Parallel decoding without losing joint structure.** The read-out tokens are drawn from a joint over the latent, not from independent marginals — so Sudoku-style constraints are satisfiable in one pass.
- **Strong structured-reasoning numbers.** Puzzle accuracy on Sudoku and Countdown improves at matched size vs discrete and continuous diffusion baselines; LM1B generative perplexity also improves.

## Gotchas & tricks

- **Token-head capacity.** The read-out head is a bottleneck; too small ⇒ the latent carries information the head can't surface. Scale with the denoiser.
- **Scaffold drift.** Early-step tokens are low-quality; feeding them back aggressively can derail the latent. The paper uses soft (distributional) conditioning rather than hard token substitution.
- **Not autoregressive, not fully parallel.** The sequential denoising schedule is the compute floor; HC-DLM is cheaper than AR for the same length but not O(1) in diffusion steps.
- **Sampling schedule matters.** Standard DDPM-style schedules work but are suboptimal; the paper mentions schedule tuning as a sensitivity axis.
- **Open-domain scaling** is unproven at the time of writing — LM1B is small; extrapolating to 100B-token pretraining is an open question.

## Sources

- Paper: *Hierarchical Continuous Diffusion Language Models* — Ren, Li, Liu, Liu, Schwing, UIUC, 2026 — [arXiv:2610.02193](https://arxiv.org/abs/2610.02193).
- Related: *MDLM / LLaDA / SEDD* — discrete diffusion LM baselines; *Diffusion-LM / Plaid* — continuous diffusion LM baselines.
