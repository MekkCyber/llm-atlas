# Orthrus (hybrid AR + diffusion speculative decoding)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Orthrus is a **hybrid autoregressive-diffusion architecture** that accelerates decoding of a frozen AR backbone by generating multiple tokens in parallel through a diffusion head, then reconciling them with the AR model via an **intra-model consensus mechanism**. The original paper claims the reconciliation is *lossless* — the accepted output sequence equals what the AR backbone alone would have produced. A 2026 independent reproduction shows that this equivalence holds under FP32 but breaks under BF16: exact-trajectory match occurs in only ~45% of prompts under BF16 across 12 domains, while downstream benchmark scores stay flat.

**Prereqs:** [_speculative-decoding](_speculative-decoding.md)
**Related:** [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

A speculative-decoding-family method where the **drafter is a diffusion head grafted onto the AR target** and the **verifier is the same AR target's own forward pass**, tied together by a consensus rule that decides which drafted tokens survive. Because the drafter and verifier share weights (the AR backbone is frozen; the diffusion head sits alongside it), Orthrus does not need a paired external draft model.

The consensus mechanism claims that its acceptance rule preserves the AR output distribution exactly, hence "lossless speculative decoding" — the produced sequence is claimed to be identical to what the AR backbone alone would emit token-by-token.

## How it works

At a high level (the reproduction paper reports on the claim, not a step-by-step redesign):

1. The diffusion head produces $k$ candidate tokens in parallel from the current hidden state.
2. The AR backbone runs one forward pass over those $k$ positions, giving its own logits.
3. The consensus mechanism compares the diffusion-drafted tokens with the AR distribution at each position under a rule that is claimed to preserve the AR marginals: on positions where consensus holds, the drafted tokens are accepted; otherwise the AR model's own sample is used.
4. Decoding continues from the last accepted position.

The claim that this yields the same output *sequence* as the AR backbone rests on the consensus rule being distribution-preserving *and* on all intermediate arithmetic being exact. That second condition is the one the 2026 reproduction stresses.

## Why it matters

- **Structural gain vs draft-model speculative decoding.** No paired external draft model is needed; the diffusion head shares the target's parameters, so deployment memory does not double.
- **Precision-dependent losslessness is a real finding.** Under BF16, exact-trajectory equivalence to the AR reference holds in only ~45% of 1,190 prompts across 12 domains for the authors' own checkpoint (and 43% for an independent re-train). Under FP32 it holds on 100% of the same prompts. Downstream lm-eval-harness benchmarks do not show systematic degradation, so the drift does not surface where teams usually look.
- **Correlation with reference-model perplexity.** Prompts on which the AR reference has higher response-conditional perplexity are the ones where trajectory divergence is most likely. Easy prompts still match; hard prompts drift.

## Gotchas & tricks

- **"Lossless" needs a precision annotation.** Any reproduction or deployment pitch for Orthrus (or any consensus-based speculative-decoding method) should state the arithmetic precision at which losslessness was measured. Under BF16 it is empirical: mostly-preserved, not exactly-preserved.
- **Downstream benchmarks are not a substitute for trajectory-equivalence tests.** The reproduction paper's headline is precisely this — lm-eval-harness scores are flat while trajectories diverge on 55% of prompts. If exact reproducibility matters (regression tests, deterministic serving), run a trajectory-equivalence check.
- **Hard prompts are where it drifts.** If the deployment mix is skewed toward long-context, high-perplexity generations (reasoning, code, long-form writing), expect more drift than an average-domain benchmark suggests.
- **FP32 fixes it but is not the deployment target.** FP32 verification is a diagnostic, not a serving mode. Practical solutions are (i) accept the drift and treat Orthrus as "downstream-lossless", (ii) fall back to AR on high-perplexity regions, or (iii) run consensus arithmetic in higher precision than the rest of the stack.

## Sources

- Paper: *How Lossless Is Lossless Speculative Decoding? The Role of Numerical Precision in Orthrus* — Ilya Koziev, Leonid Sinev, Ivan Oseledets — 2026 — [arXiv:2609.15504](https://arxiv.org/abs/2609.15504) — the independent reproduction.
- Background: the original Orthrus paper (cited within the 2026 reproduction) introduces the hybrid AR-diffusion architecture and the consensus mechanism.
