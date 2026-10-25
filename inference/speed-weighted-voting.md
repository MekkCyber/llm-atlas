# Speed-weighted voting
*Depth — using diffusion-LM decoding compute as a free confidence signal for self-consistency.*

**TL;DR:** Diffusion language models don't use a fixed number of forward passes per token: on the *same* math problem, wrong candidate answers take more passes than correct ones. Replacing uniform-weight majority voting with weighting by inverse decoding compute improves accuracy by up to 4.2 points at no extra inference cost. It's a free confidence signal that is only available to iterative decoders.

**Prereqs:** [_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [attention.md](../fundamentals/attention.md)

---

## What it is

A decoding-time ensembling rule for diffusion LMs. Given `k` sampled candidate answers to the same prompt, each candidate is tagged with the number of forward passes consumed to produce it. The final answer is chosen by weighted vote, with weight inversely tied to pass count. Expensive candidates — the ones the model found hard — count less.

## How it works

1. Sample `k` candidates from the diffusion LM with its native iterative decoder.
2. For each candidate, record `n_i` = forward passes consumed (depends on the per-token speed schedule the model adaptively chose).
3. Compute weight `w_i = f(1 / n_i)` — the paper uses a simple inverse or inverse-exponent of pass counts.
4. Vote over the candidate space (answer string, final answer token, canonicalized form) with weights `w_i`. Return argmax.

The weight is already a byproduct of decoding, so there is no additional forward pass vs plain self-consistency.

## Why it matters

- **Free confidence.** Autoregressive LMs use log-probs; diffusion LMs can use decoding speed. This is a new, decoder-internal signal with no inference overhead.
- **+4.2 points over majority voting** on the paper's benchmarks, at the same sample budget.
- **Exposes a tokenization-fairness issue.** The same measurement shows highly tokenized languages (e.g. agglutinative or non-Latin scripts) need more passes per token on the *same* content. Multilingual users pay more compute per answer — a cost-equity problem for serving.
- **Generalizes.** Any iterative decoder (consistency models, flow matching with adaptive step count) can expose a similar signal.

## Gotchas & tricks

- Pass-count variance is dominated by prompt difficulty at low `k`. Below `k ≈ 5` the speed signal is noisy; the gain over majority voting shows up in the 10–64 range.
- Weight function matters less than its monotonicity. Both inverse and inverse-exponent beat uniform; the exact form is a tuning knob.
- Not applicable to autoregressive decoders — their per-token pass count is fixed by architecture.
- Doesn't fix wrong-answer frequency; it rebalances the vote. If all `k` candidates are wrong, speed weighting can't help.

## Sources

- Paper: *Diffusion Language Models Change Speed Across Tasks and Languages* — Fayyaz, Modarressi, Israel, Bandarkar, Peng (UCLA NLP), 2026 — [arXiv:2610.25835](https://arxiv.org/abs/2610.25835)
