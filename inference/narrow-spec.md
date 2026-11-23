# NarrowSpec
*Depth — a speculative-decoding drafter whose hidden width can be 2–4× narrower than the target.*

**TL;DR:** Speculative-decoding wisdom was to match the drafter's hidden dim to the target's so the drafter inherits the target's representation. NarrowSpec shows that acceptance is dominated by the drafter's *vocabulary-aligned output* head, not its interior width. Halving or quartering the hidden dim preserves most of the acceptance rate, and for DFlash-style drafters the output projection can be **folded into the target's existing vocab projection** — no extra up-projection at inference. End-to-end speedups at batch size 1 on B200 and DGX Spark.

**Prereqs:** [_speculative-decoding](_speculative-decoding.md)
**Related:** [../pre-training/mtp](../pre-training/mtp.md)

---

## What it is

A drafter-architecture change that keeps the training/verification protocol unchanged. Instead of a draft transformer with hidden dim $d = d_{\text{target}}$, use $d_{\text{draft}} \in \{d/2, d/4\}$. Two sub-recipes:

- **General narrow drafter.** Works with any drafter family (EAGLE-3, DFlash). Trade a few points of acceptance for substantially cheaper per-draft cost.
- **DFlash-with-folded-head.** When the drafter shares the vocab projection with the target, NarrowSpec eliminates the width-mismatch adapter by **absorbing** the drafter's $d_{\text{draft}} \to V$ projection into the target's $d_{\text{target}} \to V$ matrix along the subspace the drafter writes to.

## How it works

Standard draft block (full width):

```
h = Attn(h) ; h = FFN(h)             # all at d_target
logits = h @ W_out_target            # W_out_target : d_target x V
```

Narrow draft block:

```
h_narrow = Attn_narrow(h_narrow)     # d_draft < d_target
h_narrow = FFN_narrow(h_narrow)
logits = h_narrow @ (W_up @ W_out_target) = h_narrow @ W_folded
```

where `W_up` lifts $d_{\text{draft}} \to d_{\text{target}}$ and is composed with the target's vocab head offline, producing a single matrix `W_folded : d_draft x V`. At inference, the drafter never materializes an up-projected activation. The operation count per draft token drops roughly in proportion to $d_{\text{draft}} / d_{\text{target}}$.

Acceptance rates are reported close to full-width on Qwen3-8B and GPT-OSS-20B when used with DFlash and EAGLE-3 drafters. End-to-end speedups come from the reduced drafter FLOPs at batch size 1.

## Why it matters

- **A new architectural lever in spec decoding.** Previous gains (EAGLE-3, DFlash) came from better feature passing and tighter training. Narrowing the drafter is orthogonal and composes with them.
- **Especially impactful for low-batch inference.** At batch size 1 (chat UIs, coding agents, single-user clients) the drafter contributes a large fraction of per-token wallclock. Shrinking it is directly felt in latency.
- **Deployment-friendly.** The folded vocab-head case trades zero inference compute for a one-time offline merge; no runtime complexity added.

## Gotchas & tricks

- **Not free below $d/4$.** Acceptance rates degrade past 4× narrowing in most settings.
- **Folding applies only when drafter and target share the vocab projection.** For independent-head drafters (vanilla small-model spec) the narrowing still helps but the folding trick doesn't.
- **Verifier side is unchanged.** NarrowSpec is a drafter-only change; the target model's forward and the accept/reject sampling are identical.
- **Tokenizer alignment.** Drafter and target must share the tokenizer (true in all mainstream spec-decoding setups, but worth checking).
- **Interacts with EAGLE-3 features.** Feeding the drafter with the target's last-layer features is still beneficial; narrowing the drafter's *internal* dim doesn't conflict with that, because the feature projection is on the input side.

## Sources

- Paper: *NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding* — 2026 (NVIDIA).
- Paper: *EAGLE-3* — Li et al., 2025.
- Paper: *DFlash* — the drafter family most compatible with the folded-head variant.
