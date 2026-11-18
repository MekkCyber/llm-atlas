# Speculative Decoding

*Taxonomy — the family of inference methods that propose multiple future tokens cheaply and then verify them with the full model.*

**TL;DR:** Autoregressive decoding is slow because each token waits for a full forward pass. Speculative decoding breaks that serial dependency: a **drafter** cheaply proposes a short continuation (several tokens, or a tree of candidates), and the **verifier** — the target model — scores all of them in one parallel forward pass, accepting the longest prefix that matches the target's distribution. The drafter can be a smaller model, extra heads on the target, a shallower pass of a looped model, or a learned next-*k*-token predictor. The current default for production servers is tree-based speculation (EAGLE-style) over a draft model distilled from the target.

**Related taxonomies:** none yet.
**Depth files covered here:** [loopfill](loopfill.md) · others as they land.

---

## The problem

Decoding is **memory-bound**, not compute-bound. A single GPU step moves the full weight matrix through SRAM just to produce one token. Autoregression serializes those steps, so throughput is capped by the per-step latency times the number of tokens generated. Yet the forward pass is well under 100% utilized — plenty of parallel compute is wasted waiting. Speculative decoding turns that idle compute into additional candidate tokens.

The constraint every variant is fighting: the drafter has to be *cheap* (otherwise it costs more than it saves) *and* its distribution has to match the target's often enough (otherwise the acceptance rate collapses and the saved time evaporates in rejected tokens).

---

## The shared pattern

Every speculative decoding scheme looks like:

```
1. Draft:    produce K candidate continuations (a chain, tree, or lattice).
2. Verify:   one parallel forward pass of the target scores all candidates.
3. Accept:   keep the longest prefix whose per-step distribution the target agrees with.
4. Sample:   from the target's distribution, produce one additional "bonus" token.
5. Loop.
```

Variants differ on (a) *what* the drafter is, (b) *how many* candidates and in what shape (chain, tree, lattice), (c) *how* acceptance is defined (strict equality, Leviathan rejection sampling, speculative sampling with KL correction), and (d) whether any training is needed to align the drafter to the target.

---

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Draft model (Leviathan/Chen 2023) | small LM drafts a chain; verify with the target | needs a well-matched small model | general-purpose serving with a size pair available |
| Medusa | extra decoding heads on the target predict the next *k* tokens | adds parameters and training | target can absorb extra heads; no second checkpoint |
| EAGLE / EAGLE-2 | lightweight head over the target's own features; tree draft | best acceptance rates reported | production serving where the head can be trained once |
| Lookahead decoding | n-gram-based greedy draft from the running output | no training; smaller gains | long-tail scenarios where no drafter is available |
| [loopfill](loopfill.md) | dual-axis speculation: multi-token suffix × shallow-loop depth for looped Transformers | only applies to looped models | recurrent/looped LLMs (Ouro, Huginn) at inference |

---

## How to choose

For a standard autoregressive LLM in production serving, the modern default is a trained draft head of the **EAGLE** family — it gives the best speedup-per-complexity and no second checkpoint to maintain. If you cannot train a drafter and must stay training-free, lookahead decoding is a safer pick than nothing. For **looped** reasoners (energy-based, recurrent), the token axis is only half of the available parallelism; use LoopFill-style dual-axis speculation instead.

Combine speculative decoding with continuous batching: the verify step can be batched across requests, so the "one parallel forward pass" amortizes almost for free. Speculative decoding and FP8/FP4 serving stack cleanly; the drafter and verifier can live at different precisions.

## Adjacent but distinct

- **Multi-token prediction (MTP)** — a *training* objective that teaches the main model to emit multiple tokens per step. Can be combined with speculative decoding (use the MTP heads as the drafter) but is architecturally different.
- **Parallel sampling / majority voting** — runs independent samples at the same step; not speculation, since nothing verifies.
- **Continuous batching** — interleaves requests in one forward pass; orthogonal, and complementary.

---

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023 — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., DeepMind, 2023 — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318).
- Paper: *Medusa: Simple LLM Inference Acceleration* — Cai et al., 2024 — [arXiv:2401.10774](https://arxiv.org/abs/2401.10774).
- Paper: *EAGLE / EAGLE-2* — Li et al., 2024 — [arXiv:2401.15077](https://arxiv.org/abs/2401.15077), [arXiv:2406.16858](https://arxiv.org/abs/2406.16858).
- Paper: *LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers* — Zhu et al., 2026 — [arXiv:2611.19649](https://arxiv.org/abs/2611.19649).
