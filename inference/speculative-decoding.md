# Speculative Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Run a small, fast **draft model** to propose several next tokens at once, then verify them in a **single** forward pass of the big target model. Accepted tokens commit; the first rejected token is sampled fresh from the target distribution and the rest are discarded. The scheme is **mathematically identical** to sampling from the target model alone (unbiased), while yielding 2–3× end-to-end decode speedups when draft and target agree often.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md), [speculative-forcing.md](./speculative-forcing.md)

---

## What it is

Autoregressive decoding generates one token per target forward pass — that's memory-bandwidth bound (each step must stream the KV cache). Speculative decoding breaks the one-token-per-step bottleneck by **batching speculative candidates** into one target forward pass.

The classical scheme (Leviathan et al. 2022; Chen et al. 2023) uses:

- A **small draft model** $q$ (much cheaper per forward pass than target $p$).
- A **verification rule** that accepts or rejects each drafted token so the final sample is distributed exactly as if we had sampled from $p$ directly.

Variants (Medusa, EAGLE, Lookahead) replace the external draft model with a lightweight draft *head* on top of the target itself, trading model simplicity for integration.

---

## How it works

### 1. Draft

Starting from the current context, run $k$ steps of the draft model:

$$
y_1 \sim q(\cdot | x), \; y_2 \sim q(\cdot | x, y_1), \; \ldots, \; y_k \sim q(\cdot | x, y_{<k})
$$

Typical $k = 4$ to $k = 8$.

### 2. Verify

Run **one** target forward pass that scores all $k$ drafted positions in parallel, obtaining $p(\cdot | x, y_{<i})$ for $i = 1, \ldots, k+1$.

### 3. Accept / reject

For each drafted token $y_i$ compute the acceptance probability

$$
\alpha_i = \min\!\left(1, \frac{p(y_i | x, y_{<i})}{q(y_i | x, y_{<i})}\right)
$$

and accept with probability $\alpha_i$. If accepted, commit $y_i$ and move on. On the first rejection: sample a replacement from the **adjusted** distribution

$$
p'(y) \propto \max(0, p(y | x, y_{<i}) - q(y | x, y_{<i}))
$$

commit it, and discard $y_{i+1}, \ldots, y_k$. Also commit the "free" bonus token sampled from $p$ after the last accepted position.

The acceptance-rejection scheme is a standard rejection-sampling construction; it guarantees the committed sequence has the same distribution as directly sampling from $p$.

---

## Why it matters

- **2–3× decode throughput** typical for well-matched draft/target pairs, with no quality change (same sampling distribution).
- **Zero fine-tune required.** The vanilla scheme works with any pretrained draft + target pair.
- **Composes with KV cache and continuous batching.** Lives inside the decoder loop and doesn't change tensor shapes elsewhere.
- **Design pattern.** "Draft many, verify in parallel, keep sampling honest" has become a general template: EAGLE applies it with embedding-head drafts, Medusa with multi-head drafts, and recently **speculative forcing** adapts it to the *denoising* axis of AR video generators ([speculative-forcing.md](./speculative-forcing.md)).

---

## Gotchas & tricks

- **Acceptance rate dominates speedup.** If the draft model agrees with the target on 70% of tokens, each verification pass commits ~1 + 0.7 + 0.49 + … ≈ 3 tokens on average. If agreement is 20%, speedup collapses.
- **Pick the draft carefully.** Same tokenizer, same chat template, same system prompt. Mismatched draft/target distributions kill acceptance.
- **Temperature matters.** Greedy decoding (temperature = 0) collapses to a "match or reject" rule and is the fastest regime. Higher sampling temperatures reduce acceptance.
- **Self-draft variants (Medusa, EAGLE).** Share base weights, avoid shipping two models. Need a small training phase for the draft head.
- **Not a quality lever.** Correctly implemented, speculative decoding is a pure speedup — if you see quality drops, there's an implementation bug or the acceptance rule has been relaxed.

---

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias, Google, 2022/23.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., DeepMind, 2023.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024.
- Paper: *Speculative Forcing for Efficient Autoregressive Video Generation* — 2026 — arXiv [2610.22347](https://arxiv.org/abs/2610.22347).
