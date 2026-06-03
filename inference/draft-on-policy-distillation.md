# Draft-OPD — On-Policy Distillation for Speculative Draft Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard speculative-draft training is SFT on target-generated trajectories — but at inference the drafter is evaluated on *its own* proposed blocks. The mismatch makes SFT plateau quickly. Draft-OPD fixes this with on-policy distillation: use **target-assisted rollout** to keep continuations stable, then **replay drafting at verification-exposed error positions** so the target supervises the drafter on draft-induced states. Over 5× lossless acceleration on reasoning models, +23% over EAGLE-3, +13% over DFlash.

**Prereqs:** [_speculative-decoding](_speculative-decoding.md)
**Related:** [domino-drafting](domino-drafting.md), [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Draft models (EAGLE-3, DFlash, Domino) are typically trained by SFT on `(prefix, target's next token)` pairs gathered from the target model rolling out by itself. At inference, the drafter sees *its own* recently-drafted tokens as prefix — and these tokens are sometimes wrong (that's why verification exists). The drafter has never been trained on those out-of-distribution prefixes, so its acceptance length plateaus well below what the architecture allows. The standard fix in RL — on-policy data — is hard to apply to draft models, because (a) drafters can't reliably roll out long sequences alone, and (b) if the target *fully* drives the rollout, the resulting prefixes follow the target's distribution, not the drafter's, eliminating the on-policy signal.

Draft-OPD threads this needle: continuations are target-driven (stable), but the drafter is *forced to draft* exactly at the positions where verification would have rejected its previous proposals.

---

## How it works

### Pipeline

```
1. Target M_t rolls out a sequence y_{1:T}, recording verification outcomes
   as if M_d were drafting alongside.

2. At every "error position" t where the drafter would have proposed a
   token rejected by M_t:
     - rewind to the prefix y_{<t}
     - have the drafter propose its block from y_{<t} (its own policy)
     - record M_t's logits at each draft position as the supervision target

3. Train M_d on these draft-induced (state, target-logit) pairs with a
   distillation loss (KL on accepted positions, plus a correction signal
   on rejected positions).

4. Continuations between error positions remain target-generated for stability.
```

### Why both pieces are needed

- **Target-assisted rollout alone:** prefixes are clean and on-distribution for $M_t$, but the drafter never trains on its own error tail. Same as SFT.
- **Pure drafter rollout alone:** drafter goes off-the-rails after a few errors; later positions train on nonsense.
- **Error-position replay:** focuses gradient on the exact draft-induced states where speculative verification would have rejected the proposal — i.e. the regime that actually limits acceptance length.

The supervision is the target model's *full distribution* at each replayed draft position (not just the argmax), so the drafter learns calibrated proposals on both accepted *and* rejected positions.

---

## Why it matters

- **Closes the SFT-plateau gap.** Pre-Draft-OPD, EAGLE-3 and DFlash hit a ceiling around 4–5 acceptance length on Qwen3. Draft-OPD lifts this by +13% (DFlash baseline) to +23% (EAGLE-3 baseline) with no architecture change.
- **5× lossless acceleration** on "thinking" (long-CoT) models across diverse tasks — the regime where draft quality matters most and where the SFT-plateau bites hardest.
- **Architecture-agnostic.** The recipe is a training-data construction technique; it composes with EAGLE-style autoregressive drafters, DFlash-style parallel drafters, and [Domino-style](domino-drafting.md) hybrid drafters.
- **Articulates a generic mismatch.** The "offline-to-inference" gap Draft-OPD addresses is a special case of behavior cloning's well-known compounding-error problem. Same fix (DAgger-style on-policy data) ports.

---

## Gotchas & tricks

- **Don't drop the target-assisted backbone.** Replacing it with full drafter rollouts produces noisy supervision after a few errors; the paper's ablations show this is worse than SFT.
- **Replay only at error positions.** Training on every position adds compute without lifting acceptance; focusing on error positions concentrates the gradient on the bottleneck regime.
- **Distillation loss, not next-token CE.** Use the full target distribution as supervision (forward KL) — calibration of drafter logits is what determines acceptance, and CE on hard targets under-trains it.
- **Compose with Domino-style architectures.** The base-anchored curriculum (Domino's training recipe) is orthogonal — it stabilizes the architecture; Draft-OPD stabilizes the data distribution.
- **Cost.** Generating replay data requires running the target model at every error position; budget accordingly. For large targets this is the limiting cost of the pipeline.

---

## Sources

- Paper: *Draft-OPD: On-Policy Distillation for Speculative Draft Models* — Lei, Li, Zhang, Cheng, Qu, Cui, Zhou, Ding, Luo, Cheng — SJTU / Shanghai AI Lab / Tsinghua / CUHK, 2026 — [arXiv:2605.29343](https://arxiv.org/abs/2605.29343).
- Background: *EAGLE-3* — Li et al., 2025 — SFT-trained autoregressive drafter, the canonical SFT-plateau case study.
- Background: *DAgger* — Ross, Gordon, Bagnell, 2011 — the on-policy imitation-learning ancestor.
