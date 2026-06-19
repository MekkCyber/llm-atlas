# Uniform Diffusion Language Models (UDLM)

*Depth — diffusion LMs where any token can be updated at any step, contrasting with masked diffusion's monotonic unmask schedule.*

**TL;DR:** A diffusion language model trains a denoiser that, given a partially-corrupted sequence, predicts a less-corrupted one. **Masked diffusion** corrupts by replacing tokens with a `[MASK]` symbol and unmasks monotonically — a token, once chosen, is fixed. **Uniform diffusion** corrupts by replacing tokens with random vocabulary entries, and at each denoising step *any* token can be revised — giving more flexible generation in principle. Sumi (2026) is the first fully open scratch-pretrained UDLM at scale: 7B parameters, 1.5T tokens, weights + checkpoints + data recipe released.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [transformer-block](transformer-block.md)

---

## What it is

One of the three main paradigms for non-autoregressive language modeling (alongside autoregressive and masked diffusion). UDLMs put forward distribution = uniform over vocabulary on each corrupted token; this design allows the reverse process to overwrite previously generated tokens at any step.

## How it works

**Forward process** (corrupts data to noise): for each token position at time $t$, with probability $\beta_t$ replace the token with a uniformly random one. As $t \to T$, the sequence approaches the uniform distribution.

**Reverse process** (the model): predicts the original token distribution at each position given the corrupted sequence and timestep. Standard denoising objective; cross-entropy loss over all positions.

**Sampling.** Start from a fully random sequence. At each reverse step, the denoiser outputs a per-position categorical distribution; sample to get the next iterate. Iterate $T$ steps to convergence.

**Why uniform vs masked.** Masked diffusion's $[\texttt{MASK}]$ token gives the denoiser a clean signal of "which positions need attention," but the unmask is monotonic — a wrong commitment early can't be revised. UDLMs let revision happen any time, in principle handling long-range coherence and global revisions better; but they lose the masked-position signal, so the denoiser has to discover where revision is needed.

## Why it matters

- **Reference artifact at scale.** Before Sumi, UDLMs had no public model where the community could study scaling behavior, generation dynamics, or trade-offs. Autoregressive (Llama, Mistral) and masked diffusion (LLaDA, MDLM) had multiple open checkpoints; uniform diffusion had none.
- **Competitive with AR at comparable token budgets** on knowledge / reasoning / coding benchmarks, per Sumi's evaluation. Underperforms on commonsense, attributed by the authors to the education-heavy data mixture (not necessarily a UDLM intrinsic).
- **Native bidirectional structure.** UDLMs revise tokens at any position at any step — a property useful for editing, infilling, and constrained generation that AR models have to graft on.
- **Full recipe released.** Sumi publishes the data mixture spec over public corpora, all checkpoints, and hyperparameters — enabling clean reproduction and ablation.

## Gotchas & tricks

- **Step count vs quality.** More reverse steps = better samples (up to a plateau) but linear inference cost. UDLMs are competitive at AR's token budget in *training*, but inference is currently more expensive.
- **No clear "next token" — the denoiser predicts everything everywhere.** Sampling strategies (per-step temperature, top-k masking, partial-decoding tricks) are still an open design space.
- **Data mixture matters more than for AR.** With every position predicted at every step, the loss spreads thin across the sequence; data quality and diversity have outsized impact.
- **Watch for the commonsense gap.** Sumi's underperformance there is a data-mixture story per the authors; reproductions should test other mixtures before concluding UDLMs are weaker on commonsense.

## Sources

- Paper: *Sumi: Open Uniform Diffusion Language Model from Scratch* — Kudo, Ikeda, Matsuda, Sakaguchi, Suzuki, Tohoku University, 2026 — [arXiv:2606.19005](https://arxiv.org/abs/2606.19005).
- Related: *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM) — Austin et al., 2021 — original uniform-vs-masked discrete-diffusion framing.
- Related: *Simple and Effective Masked Diffusion Language Models* (MDLM) — Sahoo et al., 2024 — masked-diffusion counterpart for comparison.
