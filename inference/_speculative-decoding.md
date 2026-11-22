# Speculative Decoding

*Taxonomy — accelerate autoregressive decoding by proposing multiple tokens cheaply, then verifying them in a single target-model forward.*

**TL;DR:** The target model is bottlenecked by sequential decode steps. A cheaper **drafter** proposes `k` tokens ahead; the target runs one forward over all `k+1` positions in parallel and accepts the longest prefix that matches a sampled realization of its own distribution. Acceptance is **lossless in distribution** — the final output is a correct sample from the target. The family splits along three axes: what the drafter is, how the drafter is trained or attached, and how acceptance is computed.

**Related taxonomies:** *(none yet in `inference/`)*
**Depth files covered here:** [narrow-draft-sd](narrow-draft-sd.md) · [vla-speculative-decoding](vla-speculative-decoding.md)

---

## The problem

Each decoded token requires a target forward; memory bandwidth, not FLOPs, dominates at low batch. If a cheap predictor can guess the next few tokens, most of them will be right, and verifying all of them in one target forward reclaims the sequential cost — for free in quality.

## The shared pattern

```
draft:   d_1, ..., d_k   = drafter(context)
verify:  for all positions, target_logits = target(context || d_1..d_k)
accept:  longest prefix where sampled target token == d_i (per Leviathan speculative-sampling rule)
emit:    accepted tokens + one bonus target-sampled token on the first mismatch
```

Every variant differs in **who is drafting**, **how much hidden state is shared with the target**, and **how acceptance interacts with the target's distribution**.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Draft-model SD (Leviathan 2022) — *no depth file yet* | A small separate LM as drafter | Serving two models; drafter training | Default when a small matching LM is available |
| EAGLE / Medusa — *no depth file yet* | Drafter is attached heads over target's hidden state | Less flexible than a separate drafter | Best acceptance rate; tight target integration |
| [**NarrowSpec**](narrow-draft-sd.md) (NVIDIA 2026) | Drafter's residual width shrinks below the target's | Small acceptance hit | Cheap further drafter shrink without losing SD gains |
| [**VSpark**](vla-speculative-decoding.md) (KAIST AI 2026) | SD specialized for autoregressive vision-language-action models | VLA-specific; need action-noise-calibrated rollouts | Robot / VLA latency-critical settings |
| Lookahead decoding — *no depth file yet* | No learned drafter; parallel multi-ngram proposals | Lower acceptance than learned drafters | When no drafter is available at all |

## How to choose

- **Have a matching small LM?** → draft-model SD.
- **Max acceptance at serving time?** → EAGLE / Medusa — attach heads to target hidden state.
- **Drafter already exists and is "too wide"?** → NarrowSpec — shrink the drafter residual.
- **Non-text modality (VLA, audio)?** → calibrate the drafter to the modality's token/noise distribution (VSpark for VLA).
- **No drafter at all, want some win?** → lookahead.

Stack doesn't apply — one drafter at a time.

## Adjacent but distinct

- **KV cache sharing** across prefill/decode — a different latency lever (prefill-decode disaggregation); not sampling-equivalent to the target.
- **Beam search** — multi-token search that changes the output distribution. SD is lossless; beam isn't.
- **Parallel decoding with blocks** (SpecExec and variants) — generalize SD to tree-shaped proposals; same family, different search structure.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2022 — the original framing + the acceptance rule.
- Paper: *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — Li et al., 2024.
- Paper: *Medusa: Simple LLM Inference Acceleration via Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding* — 2026 — [arXiv:2611.23835](https://arxiv.org/abs/2611.23835).
- Paper: *VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models* — Yoo et al. (KAIST AI), 2026 — [arXiv:2611.22666](https://arxiv.org/abs/2611.22666).
