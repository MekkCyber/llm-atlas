# Multi-Tier Speculative Decoding (Intra-Model Routing)
*Depth — confidence-routed verification across slim intra-model submodels for cheaper speculative decoding.*

**TL;DR:** Standard speculative decoding pairs a small draft with a single full-model verifier — every drafted token costs one full-model forward pass to verify, even tokens the draft is nearly certain about. **Multi-tier speculative decoding** adds intermediate verification tiers using *slim submodels carved from the target model itself*, routes each token to the appropriate tier by confidence, and only escalates to the full model for low-confidence tokens. VIA-SD (2026) is the implementation; it preserves output identity while cutting verification cost on the largest token bucket.

**Prereqs:** *(none — speculative decoding background recommended)*
**Related:** [README.md](README.md)

---

## What it is

A generalization of speculative decoding where verification cost is *adaptive*, not uniform. The setup:

- **Draft model** produces candidate tokens (as in standard SD).
- **Slim tiers** — submodels of the target carved by skipping layers, pruning heads, or quantizing — provide cheap verification.
- **Full target** — verifies only the low-confidence tail.

Each drafted token is routed to a tier based on a cheap confidence signal (draft logit margin, calibration score). Tokens with high confidence get accepted after slim verification; tokens with medium confidence escalate; only low-confidence tokens hit the full model.

---

## How it works

### Tier selection

A confidence score per drafted token (e.g. draft model's softmax margin) determines the tier. Tier thresholds are calibrated offline so that the *output distribution matches* the target model — this is the property speculative decoding needs to preserve exactness.

### Slim submodels

The intermediate tiers are submodels of the target — layer-skipped versions, pruned heads, or quantized weights. Sharing weights with the target keeps memory footprint roughly constant.

### Verification

For each tier, the standard speculative-decoding accept/reject test runs: if the slim tier's distribution covers the draft, accept; otherwise escalate. Acceptance is biased toward easier tokens, escalation toward harder ones.

### Output exactness

Provided the tier thresholds are calibrated against the target distribution, the final output distribution is identical to running the full target greedily/sampling — same correctness guarantees as standard SD.

---

## Why it matters

- **Verification cost was the next obvious target.** Speculative decoding made drafting cheap; verification was uniform-cost and the next bottleneck.
- **Confidence-adaptive computation is general.** Routing tokens by confidence is reusable across any decoding accelerator.
- **Slots into production stacks.** No new model — slim tiers are derived from the target, so deployment is a calibration + routing change.

---

## Gotchas & tricks

- **Calibration must be honest.** If the tier thresholds don't preserve output distribution, you've quietly degraded the model. Validate per-tier acceptance rates on a held-out set.
- **Tier-count tradeoff.** Two tiers (slim + full) capture most of the win; three+ tiers add complexity without proportional benefit.
- **Pathological tokens.** Tokens where the draft is confident but wrong waste slim-tier compute before escalation. Track per-token escalation rates.
- **Caching across tiers.** KV state for the slim tier and full tier must be kept consistent; serial tier execution is simpler than parallel.

---

## Sources

- Paper: *VIA-SD: Verification via Intra-Model Routing for Speculative Decoding* — Xian et al., Zhejiang U., 2026 — [arXiv:2606.12243](https://arxiv.org/abs/2606.12243).
