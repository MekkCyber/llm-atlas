# Speculative decoding
*Taxonomy — parallelize predictive work with cheap verification to accelerate autoregressive decode.*

**TL;DR:** Autoregressive decode is fundamentally sequential — one token per forward pass — which under-utilizes modern accelerators. Speculative decoding runs a cheaper **drafter** that proposes multiple tokens ahead, then verifies them in one forward pass of the full model. Any token with a high enough acceptance probability is kept; the rest are discarded and the loop resumes. Lossless when the verifier is the base model and the acceptance rule matches its distribution; approximate variants trade a small quality hit for larger speedups.

**Related taxonomies:** none
**Depth files covered here:** [speculative-listening](speculative-listening.md)

---

## The problem

Each AR decode step is a single matmul-bound forward pass, so GPU compute is idle most of the time. If we could predict multiple tokens per base-model pass — with a way to detect wrong predictions — we'd amortize the fixed per-step cost. The constraint: the full model's output distribution must be preserved (lossless) or only approximately matched in a well-controlled way (approximate).

## The shared pattern

```
draft    ──▶  propose tokens t_{n+1} … t_{n+K} from a cheaper distribution q
verify   ──▶  base model computes p(·|context) for all K candidates in ONE pass
accept   ──▶  accept each t_i with probability min(1, p/q);  stop at first reject
resample ──▶  on reject, resample from (p − q)_+ / Σ(p − q)_+
```

Every variant has (i) a drafter that is cheap but somewhat correlated with the base distribution, and (ii) a verification step that reuses the base model's parallel-prefill efficiency.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Vanilla draft-model (Leviathan et al.) | Small LM drafts, big LM verifies | Drafter must share tokenizer, train or tune separately | Clean 2–3× speedup, broadly applicable |
| Medusa | Add parallel heads on the base model to predict future tokens | Changes base model; not training-free | Same model, no extra memory |
| EAGLE / EAGLE-2 | Draft from top-layer features, not tokens | Needs retraining the draft head | Higher acceptance → bigger speedup |
| Lookahead decoding | n-gram–based drafting from recent generation | Zero training; smaller gains | Code / repetitive generation |
| Self-speculative | Partial-layer base model as drafter | Simpler deploy; moderate gains | Memory-constrained |
| [speculative-listening](speculative-listening.md) | Draft audio chunks during idle listening time | Only applies to full-duplex speech LMs | Full-duplex voice agents |

## How to choose

- **Vanilla draft-model** is the default for plain LLM serving if you have a small aligned model (same tokenizer, same post-training target).
- **EAGLE / EAGLE-2** when you can retrain a small head and want the best speedup per request.
- **Medusa** when you own the base model and can co-train heads.
- **Lookahead** when you need training-free and your workload has repetitive structure (code, JSON).
- Combine with **continuous batching** and **paged attention** — speculative decoding composes with both; the acceptance machinery works per-request.

## Adjacent but distinct

- Prefill/decode disaggregation — speedup by separating prefill and decode onto different hardware, not by predicting ahead.
- Parallel sampling (best-of-$n$) — quality play, not latency play.
- Non-autoregressive decoding — gives up the AR structure entirely; different quality tradeoff.

## Sources

- Paper: Fast Inference from Transformers via Speculative Decoding — Leviathan, Kalman, Matias, 2023 — https://arxiv.org/abs/2211.17192
- Paper: Medusa — Cai et al., 2024 — https://arxiv.org/abs/2401.10774
- Paper: EAGLE — Li et al., 2024 — https://arxiv.org/abs/2401.15077
- Paper: Decode Before You Hear: Speculative Listening — Bian et al., 2026 — [arXiv:2611.06982](https://arxiv.org/abs/2611.06982)
