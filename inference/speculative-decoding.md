# Speculative decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Accelerate autoregressive decoding by using a **small, fast drafter model** to propose several tokens at a time, then **verifying them in parallel** with the full (target) model. Accepted tokens are kept; the first rejected token is re-sampled from the target distribution. If a draft of length $k$ is accepted with probability $p$, the expected speedup over single-token decoding is $\approx \frac{1 - p^{k+1}}{(1-p)(1 + c)}$, where $c$ is the drafter's per-token cost relative to the target. The verifier's acceptance rule can be set to be **lossless** (output distribution unchanged) or **lossy** (traded for speed).

**Prereqs:** [README.md](README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [speculative-diffusion-drafting.md](speculative-diffusion-drafting.md)

---

## What it is

Standard autoregressive decoding spends one full target-model forward per token. The target model has enough parallelism to score *many* candidate tokens at once, so most of the forward is wasted. Speculative decoding fills that unused parallelism by cheaply guessing tokens, then checking all guesses in a single target forward.

## How it works

Per decoding step:

1. **Draft.** A small drafter model autoregressively samples $k$ candidate tokens $t_1, \ldots, t_k$ from the current context.
2. **Verify.** The target model does one forward over the context plus all $k$ draft tokens, producing $k+1$ probability distributions in parallel.
3. **Rejection-sample.** For each draft position $i$ in order, accept $t_i$ with probability $\min\big(1, p_\text{target}(t_i) / p_\text{draft}(t_i)\big)$. On rejection, re-sample $t_i$ from the max-corrected residual $\big[p_\text{target} - p_\text{draft}\big]_+$ and stop.
4. **Append & repeat.** Append all accepted tokens plus the (possibly re-sampled) first rejected token. One target forward may yield up to $k+1$ output tokens.

The rejection-sampling rule above is **lossless** — the distribution of outputs is identical to vanilla sampling from the target model. Relaxing the rule (accept more aggressively) gives a **lossy** variant that trades quality for speed.

### Choosing the drafter

Common choices: (a) a dedicated small draft model (OpenAI's original paper, DeepMind's work), (b) early-exit heads on the target model (Medusa, EAGLE), (c) a diffusion LM used as a parallel drafter (see [speculative-diffusion-drafting.md](speculative-diffusion-drafting.md)). The right choice depends on how closely the drafter can mimic the target's distribution per unit compute.

## Why it matters

- **Latency win is substantial.** On typical LLM workloads, 2–3× end-to-end speedups are common with no quality loss; EAGLE/Medusa variants push further.
- **Doesn't change the target model.** The target weights are untouched, so speculative decoding composes with quantization, batching, KV-cache tricks, and most other inference optimizations.
- **Clean lossless/lossy knob.** The acceptance rule gives operators an explicit quality-vs-latency control, which few inference tricks offer this cleanly.

## Gotchas & tricks

- **Draft/target divergence kills gains.** If the drafter's distribution is far from the target's, acceptance probability drops and the extra drafter cost dominates. Draft-target alignment (via distillation or shared architectures) is usually necessary.
- **Variable $k$ matters.** A fixed draft length $k$ is suboptimal when acceptance varies with context. Adaptive $k$ (grow when accept rates are high, shrink when low) recovers much of the lost speedup.
- **Memory bandwidth, not FLOPs, is the usual bottleneck.** The target forward over $k+1$ tokens is nearly free in compute but still pays the full weight-load cost. Speculative decoding wins because of this, not despite it.
- **Lossy acceptance can be dangerous.** Even small acceptance-rule relaxations shift the output distribution; measure quality end-to-end, not just acceptance rate.
- **Batched serving changes the math.** In a batched inference stack, the target forward is already shared across requests, so marginal speculative speedups shrink. The win is largest for low-batch / single-stream latency.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., Google, 2023 — https://arxiv.org/abs/2211.17192
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., DeepMind, 2023 — https://arxiv.org/abs/2302.01318
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024.
