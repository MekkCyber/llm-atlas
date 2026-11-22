# VSpark — speculative decoding for VLA models

*Depth — speculative decoding tailored to autoregressive vision-language-action (robot) models.*

**TL;DR:** Autoregressive VLA models emit robot actions one token at a time, so the usual decode bottleneck applies — but a text-trained drafter wastes acceptance because action-token distributions are shaped by proprioceptive and visual features, not language. VSpark adds three changes: train the drafter on rollouts with **calibrated action-space noise**, align the drafter's hidden features to the VLA's, and **reuse predictions from rejected drafts** instead of discarding them. Lossless 1.28×–2.10× end-to-end speedup across four VLAs and four manipulation benchmarks.

**Prereqs:** [_speculative-decoding.md](_speculative-decoding.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A VLA-specific instantiation of speculative decoding. VLA policies tokenize continuous actions and predict them autoregressively, usually conditioned on image frames and language instructions. Each action step has a decode cost comparable to a short LLM generation — but in a latency-sensitive loop where the robot is waiting for the next command.

## How it works

Three additions on top of standard draft-model SD:

1. **Action-noise-calibrated rollouts.** The drafter is trained on rollouts collected with noise injected in action space, matched to the target VLA's realistic stochasticity. This surfaces the modes the drafter must learn to track.
2. **Feature alignment.** An auxiliary loss ties the drafter's hidden representations to the target VLA's hidden features at matching layers. Acceptance becomes more predictable because the drafter is trained to live in the same feature geometry as the target.
3. **Rejected-draft reuse.** When the target rejects a drafter token at position `i`, the remaining positions `i+1..k` are not discarded — their draft-side predictions feed into the next drafting round as prior context, amortizing their compute.

The acceptance rule is the standard Leviathan one; losslessness is preserved.

## Why it matters

As VLA models get deployed in real robots, decode latency per action tops the serving budget. VSpark is the first SD recipe that treats VLA tokens as *their own modality* — not reusing a text drafter. Opens the door to modality-specialized SD for audio and video action models.

## Gotchas & tricks

- The noise distribution has to match the deployment setting; drafter trained with the wrong noise shape underperforms a text-style drafter.
- Feature alignment is implemented as an auxiliary loss during drafter training; strength has to be tuned so it doesn't overwhelm the next-token objective.
- Rejected-draft reuse helps most when `k` is large (`k ≥ 4`); at `k = 2` it's a small gain.
- Ties to specific VLA tokenization schemes — needs re-derivation for VLAs with action heads that aren't token-shaped.

## Sources

- Paper: *VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models* — Yoo, Jang, Cho, Lee, Jeong, Chung (KAIST AI), 2026 — [arXiv:2611.22666](https://arxiv.org/abs/2611.22666).
