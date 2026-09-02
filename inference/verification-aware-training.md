# Verification-Aware Training (VAT)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A draft-model training recipe for [speculative-decoding.md](speculative-decoding.md) that teaches the draft to propose tokens the target will *accept*, not just tokens that minimize next-token loss. Adds a lightweight binary **verification head** (predicts "will the target accept this token?") and adaptive per-position loss weighting driven by observed rejection patterns. Up to +11.4% acceptance length and +8.7% wall-clock speedup vs standard draft training, no change to the decoding algorithm.

**Prereqs:** [speculative-decoding.md](speculative-decoding.md)
**Related:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)

---

## What it is

Standard draft-model training minimizes cross-entropy against target-model outputs. That optimizes for KL-to-target, but the *game* played at inference is different: the draft only benefits when the token is *accepted* by the speculative rejection-sampling test. Positions the draft systematically over-shoots or under-shoots on receive no correction from the CE loss.

VAT closes that loop at training time by (a) explicitly modeling the accept/reject event and (b) upweighting positions the model is getting rejected at.

## How it works

Two additions on top of standard draft training:

**1. Verification head.** A small binary classifier hangs off the draft's last hidden state, trained to predict whether the proposed token would be accepted by the target model. Training targets are computed offline by running the target on the same context. The head is used *at training* to modulate the loss and can also be used at inference to short-circuit unlikely-to-accept proposals.

$$
L_{\text{VAT}} = L_{\text{CE}}(x_t) + \lambda \cdot L_{\text{BCE}}(\hat{a}_t, a_t)
$$

where $a_t \in \{0, 1\}$ is the ground-truth accept event and $\hat{a}_t$ is the verification-head output.

**2. Adaptive rejection-pattern weighting.** During training, track per-position (or per-context-type) rejection rates. Upweight the CE loss on positions where the draft gets rejected more often — the draft learns most where it's currently failing.

Net effect: the draft's proposal distribution shifts toward the region where target-acceptance is high, even at some CE cost.

## Why it matters

- **Drop-in for existing spec-dec stacks.** No decoding-algorithm change. Same target model, same paged attention, same batching.
- **Attacks the right metric.** Acceptance length, not draft CE, is what determines throughput. VAT is the first training method to optimize it directly.
- **Compounds with structural improvements.** Orthogonal to Medusa-style parallel heads and EAGLE-style feature-level drafts — those change *how* proposals are generated; VAT changes *what* they're trained on.

## Gotchas & tricks

- **Verification head is data-hungry.** Needs many trajectories through the target model to get reliable accept/reject labels; expensive per training example.
- **Adaptive weighting can concentrate too aggressively.** If a small pathological context type dominates rejection statistics, the CE loss on the rest of the distribution starves. Cap the weighting range.
- **$\lambda$ tunes the CE ↔ accept-BCE tradeoff.** Too large and the draft optimizes accept probability at the cost of proposing sensible next tokens (leading to lower final quality when accepted). Paper's reported gains hold in a narrow $\lambda$ band.
- **Verification head at inference is optional.** Some deployments use it to filter obvious rejects before the full spec-dec pass — cheaper but adds a hyperparameter (the accept-probability threshold).

## Sources

- Paper: *Verification-Aware Training for Speculative Decoding* — Gu, Heo, Jun, Kang, Lee, Yun, Han — NAVER, 2026 — arxiv.org/abs/2608.30135.
