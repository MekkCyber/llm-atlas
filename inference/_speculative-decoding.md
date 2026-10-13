# Speculative Decoding

*Taxonomy — using a cheap drafter to propose multiple tokens that an expensive verifier accepts or rejects in one pass.*

**TL;DR:** Decoding one token at a time is memory-bandwidth-bound: the GPU reads the full weight matrix per token no matter how small the compute. Speculative decoding breaks that by drafting K tokens cheaply, running the verifier on all K in parallel, and keeping the longest prefix the verifier agrees with. Over the last three years the field has moved from external draft models to on-model draft heads (Medusa, EAGLE) and parallel drafting with cross-token correction (DCOCO).

**Related taxonomies:** [_number-formats](../quantization/_number-formats.md)
**Depth files covered here:** [dcoco](dcoco.md)

---

## The problem

Autoregressive decoding is bandwidth-bound: for small batch sizes the GPU spends most of its time reading weights from HBM, not computing. Each generated token costs one full forward pass over the model. If we could amortize that pass over multiple tokens, latency and throughput would both improve. Speculative decoding is that amortization.

## The shared pattern

Every variant has:

1. **A drafter** that proposes K candidate tokens cheaply (small model, lightweight head, parallel branches).
2. **A verifier** — the full model — that scores all K in a single forward pass.
3. **An acceptance rule** that keeps the longest prefix on which the verifier's distribution matches the drafter's within some tolerance (classic rejection sampling for exactness, or looser rules for speed).

Rejected tokens cost a decode step with no new output, so the *acceptance rate* is the single most important metric. Methods compete on raising it without making the drafter too expensive to run.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Vanilla speculative decoding (no depth file yet) | A separate, small draft model | Needs a trained draft model; mismatched vocab is painful | Any backbone with a matching small sibling |
| Medusa (no depth file yet) | Multiple parallel decoding heads on the backbone itself | Heads see no context from each other; weak on structured output | Backbone deployed; no spare draft model |
| EAGLE (no depth file yet) | Autoregressive draft head conditioned on hidden states, not tokens | Draft head is itself serial for K steps | High-quality drafts matter more than max speed |
| Lookahead decoding (no depth file yet) | Jacobi-style parallel decoding with no draft model | No training needed; lower peak speedup | Zero-training budget |
| [dcoco](dcoco.md) | Lightweight corrector module restores intra-block dependencies between parallel drafts | Extra pass of the corrector, but kept parallel | Medusa-style setups on structured output |

## How to choose

- **Already have a small sibling model?** Use vanilla speculative decoding — simplest, well-understood acceptance semantics.
- **Deploying one backbone, want speedup fast?** Medusa heads, especially with a corrector like DCOCO on top for structured output.
- **Max accuracy, willing to pay for training?** EAGLE.
- **Zero-training constraint (no fine-tune allowed)?** Lookahead decoding.

Across variants, the acceptance rate on *structured* output (code, JSON) is the stress test — parallel-draft methods lose there, which is the gap DCOCO plugs.

## Adjacent but distinct

- **Prompt caching / prefix caching** — reuses *past* KV; speculative decoding predicts *future* tokens. Orthogonal; both are typically on.
- **Multi-token prediction (MTP) at training time** — teaches the model itself to output multiple tokens; speculative decoding is pure inference-time.
- **Beam search** — explores multiple hypotheses; speculative decoding is single-hypothesis with a draft.

## Sources

- *Fast Inference from Transformers via Speculative Decoding* — Leviathan, Kalman, Matias — the original (ICML 2023).
- *Medusa* — multiple decoding heads on the backbone.
- *EAGLE / EAGLE-2* — draft heads conditioned on hidden states.
- Paper: *Contextual Correction of Parallel Drafts for Speculative Decoding* — Eisenstadt et al., 2026 — [arXiv:2610.12570](https://arxiv.org/abs/2610.12570)
