# Speculative decoding
*Depth — a lightweight drafter proposes tokens that the target model verifies in parallel, trading extra forward passes for wall-clock speedup.*

**TL;DR:** A small, fast *drafter* model proposes `k` next tokens; the target LLM runs one forward pass that scores all `k` positions at once and accepts the longest prefix consistent with target sampling. Accepted tokens cost ~1 target step for `k` outputs; rejected tokens fall back to a target sample. End-to-end lossless if the verification rule matches the target distribution.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [./mmflash.md](./mmflash.md)

---

## What it is

Autoregressive decoding is latency-bound by one serial target forward per token. Speculative decoding breaks the serial chain: a cheaper drafter generates a candidate continuation in `k` steps that the target scores in **one** parallel pass. The expected acceptance length controls how much latency drops.

## How it works

1. **Draft.** Run the drafter model for `k` tokens, producing proposed logits/samples `x̃_{1..k}`.
2. **Verify.** Run one target forward on `[prefix, x̃_{1..k}]`; read target logits at each of the `k` positions in parallel.
3. **Accept.** For each position `i`, accept `x̃_i` with probability `min(1, p_target(x̃_i) / p_draft(x̃_i))`. Reject at the first position `j` that fails; resample `x_j` from the residual distribution `norm(max(0, p_target - p_draft))`.
4. **Commit.** Advance by `accepted + 1` tokens; loop.

Variants change the drafter: a separate small LLM, a cut-down version of the target (self-speculative), a tree of candidates (Medusa, EAGLE), or a diffusion LM for parallel refinement (MMFlash).

## Why it matters

Decode latency is the dominant cost in interactive LLM serving. Drafters cost tens of ms per `k` tokens; the target pays one forward for an expected `~2–5` accepted tokens, so observed speedups are 1.5–4× on text, more with aggressive trees. Output distribution is exactly the target's — no accuracy tax when the verification rule is correct.

## Gotchas & tricks

- **Acceptance rate collapses** when the drafter and target disagree on hard-to-predict tokens (rare words, numeric spans, modality-grounded tokens). Visual-grounded VLM tokens are a notorious case — see MMFlash.
- **Tree drafting** (Medusa, EAGLE) explores multiple candidates per step; needs a verifier that scores all tree paths in one pass.
- **KV cache management** — accepted tokens reuse the target's cached keys/values; rejected positions must be rolled back correctly.
- **Temperature mismatch** between drafter and target hurts acceptance; align sampling temperatures or use typical decoding thresholds.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023 — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., 2023 — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024 — [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
