# Diffusion Language Models
*Depth — language models that generate text by iterative denoising over all positions in parallel, rather than left-to-right autoregression.*

**TL;DR:** A diffusion LM parameterizes the data distribution via an iterative denoising process. A corrupted sequence (noised tokens, masked tokens, or a continuous latent) is progressively refined over `T` denoising steps — each step calls the model on the *full* sequence and updates every position in parallel. Unlike autoregressive LMs that produce one token per forward pass, diffusion LMs trade sequential dependency for parallel decoding, giving large wall-clock wins on long outputs when `T` is small. Headline systems in 2026 include Dream-7B and the Tiny-A2D family; DiffResearch benchmarks them on agentic workloads.

**Prereqs:** [transformer-block](transformer-block.md), [attention](../fundamentals/attention.md)
**Related:** [mla](mla.md), [_moe](_moe.md)

---

## What it is

Autoregressive LMs model `P(x_1, …, x_n) = Π_t P(x_t | x_{<t})` and sample token by token: cost scales with sequence length because each token waits on its predecessor.

Diffusion LMs model the same distribution via a *noising* process `q(x_t | x_{t-1})` and learn the reverse *denoising* process `p_θ(x_{t-1} | x_t)`. At inference, start from pure noise (or fully masked tokens) and run `T` denoising steps, each one updating all `n` sequence positions in parallel on a single forward pass of the model.

Three concrete flavors in 2026:

- **Masked diffusion** — tokens are either fully observed or `[MASK]`; denoising predicts the next unmasked token at each step.
- **Uniform diffusion** — tokens corrupted uniformly to a random alternative; denoising reverses the uniform corruption.
- **Continuous latent diffusion** — denoise a continuous latent, then decode to tokens.

The last of these is close in spirit to image diffusion; the first two keep the signal discrete throughout.

---

## How it works

Training objective (schematically):

```
sample t ~ Uniform(1, T)
corrupt: x_t = q(x | x_0, t)
loss = CE( model(x_t, t), x_0 )  # predict x_0 (or next x_{t-1}) at every position
```

Decoding at inference:

```
x_T = pure noise / all [MASK]
for t = T, T-1, …, 1:
    x_{t-1} = sample from p_θ(x_{t-1} | x_t)
return x_0
```

Each denoising step is one forward pass over the whole sequence. Total compute is `T × n²` for attention (same shape as autoregressive attention, but called `T` times instead of `n` times) — so diffusion wins when `T < n`.

Modern diffusion LMs use `T` in the dozens to low hundreds. For long outputs (research reports, essays), `T` can be smaller than the output length, giving a wall-clock speedup even though per-step cost is higher.

---

## Why it matters

Autoregressive LMs will stay the default when the output is short or the model is small; the per-token overhead is tolerable. Diffusion LMs become compelling when the output is long and parallel decoding compounds: deep-research agents, long-form document generation, code projects. DiffResearch demonstrated ~6× end-to-end speedup on 100 Deep Research Bench tasks by swapping an autoregressive core for a diffusion one.

A less-discussed property: the *intermediate denoising state* is a semantically-meaningful partial draft. DiffResearch's "fast decomposition" mode reads intermediate states to produce an early plan — an operation that has no analogue in autoregressive decoding.

---

## Gotchas & tricks

- **Step budget tuning.** `T` is a quality/speed knob: too small → poor samples; too large → no speedup. Looped Uniform Diffusion LMs show the step budget can be *trained* alongside the loop budget.
- **No left-right causal mask.** Attention is bidirectional at every step; standard autoregressive KV-cache tricks don't apply directly. Specialized caches reuse computation across steps rather than across positions.
- **Decoding diverges from training.** Low-step inference can expose a train/test gap; distillation or consistency-model-style tricks (ported from image diffusion) help.
- **Instruction-tuning is less mature.** The RLHF/GRPO stack is heavily tuned for autoregressive LMs; the diffusion-LM equivalent is still catching up.

---

## Sources

- Paper: *Looped Uniform Diffusion Language Models* — Saiteja Utpala, Chieh-Hsin Lai, Yuki Mitsufuji, Praneeth Vepakomma — Sony AI / MBZUAI, 2026 — [arXiv:2610.30066](https://arxiv.org/abs/2610.30066).
- Paper: *DiffResearch: Accelerating Deep Research with Diffusion Language Models* — ITMO University, 2026 — [arXiv:2610.28909](https://arxiv.org/abs/2610.28909).
- Background: *Diffusion-LM Improves Controllable Text Generation* (Li et al., 2022); SEDD, Mercury, Dream-7B.
