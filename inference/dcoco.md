# DCOCO
*Depth — lightweight contextual correction for parallel speculative drafts.*

**TL;DR:** Parallel-draft speculative decoding (Medusa-style) runs multiple draft heads simultaneously, so each drafted token can't see the others in its block. On structured output (code, JSON) this hurts acceptance rate badly. DCOCO inserts a cheap *corrector* between drafting and verification that rewrites each position's logits conditioned on earlier positions in the block — restoring the intra-block dependencies while keeping drafting parallel. Reports up to **6.6×** speedup over vanilla decoding across four verifier backbones and ten benchmarks.

**Prereqs:** [_speculative-decoding](_speculative-decoding.md), [attention](../fundamentals/attention.md)
**Related:** (none yet)

---

## What it is

A small module trained to post-process parallel draft blocks. It takes the K draft logit vectors produced by a Medusa- or EAGLE-style head and outputs K *corrected* logit vectors in a single pass, where each position has been conditioned on the earlier positions' top-1 picks (or a short sum over their top-K distributions).

## How it works

Pipeline:

1. **Draft block.** Parallel heads produce K distributions. Each is independent.
2. **Corrector pass.** DCOCO runs attention *within the block* with a causal mask: position *i* attends to positions 0..i-1. One forward pass of a shallow transformer suffices.
3. **Acceptance.** The verifier runs on the corrected block as usual. Acceptance rates rise because the block now looks more like what the verifier would have produced sequentially.

The corrector is kept small (few layers, same hidden size as the draft head) so the added cost per block is a small fraction of a verifier pass, while the acceptance rate rises enough to pay for it several times over.

## Why it matters

Fixes the single biggest practical weakness of Medusa-style decoding — poor acceptance on structured output — without changing the verifier or giving up parallel drafting. It is a drop-in on top of existing stacks and reportedly stacks cleanly with existing draft heads across four tested backbones.

## Gotchas & tricks

- **Corrector size tradeoff.** Too small → doesn't restore dependencies; too large → eats the gains. The paper's recommended size is a shallow 2–4-layer block.
- **Needs retraining when the draft head changes.** The corrector is trained to match the verifier conditioned on this drafter's errors; swapping the drafter means retraining.
- **Still parallel-draft.** DCOCO is a correction *of* parallel drafts, not a replacement for sequential decoding — the asymptotic ceiling is set by the drafter.

## Sources

- Paper: *Contextual Correction of Parallel Drafts for Speculative Decoding* — Eisenstadt, Lidar, Gonen et al., 2026 — [arXiv:2610.12570](https://arxiv.org/abs/2610.12570)
