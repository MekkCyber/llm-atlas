# NarrowSpec — narrower-draft speculative decoding

*Depth — the drafter's residual width doesn't need to match the target's.*

**TL;DR:** Conventional speculative-decoding wisdom pairs a drafter with the same residual width as the target to keep acceptance up. NarrowSpec shows this is largely overhead — halving or even quartering the drafter residual preserves most of the acceptance and the cheaper draft forward pays back the small drop. End-to-end 1.06×–1.44× speedups over width-matched drafters across two GPUs, two drafter families, and two target models.

**Prereqs:** [_speculative-decoding.md](_speculative-decoding.md)
**Related:** [README.md](README.md)

---

## What it is

A design choice sitting under "drafter model architecture." The claim is that drafter *residual width* (the transformer's hidden size) can be shrunk independently of the target without wrecking the drafter's usefulness for speculative decoding. Depth, layer count, and other knobs are kept, but width — which dominates parameter count and compute per draft step — goes down.

## How it works

Standard draft-model SD: a small LM proposes `k` tokens, target verifies them in one forward.

NarrowSpec modifies the drafter architecture to use a **halved or quartered residual width** compared to the target, keeping the rest of the drafter stack otherwise similar. The drafter is trained conventionally (next-token prediction on the drafter's own training data; distillation from the target is optional but not required to see the gain).

Trade quantities to watch:
- **Draft forward cost** scales roughly with (width)² — halving width cuts it ~4×.
- **Acceptance rate** drops only modestly in the paper's eight configurations — small enough that the net is positive.

The acceptance rule itself is unchanged; this is purely a drafter-architecture change.

## Why it matters

Serving fleets that already run EAGLE- or Medusa- or draft-LM-style SD can shrink their drafter without retraining the target. The intervention is cheap: train a narrower drafter, swap it in, measure end-to-end latency — one of the cleanest "almost free" wins for SD since the original acceptance-rate papers.

## Gotchas & tricks

- The right width ratio depends on the target. Halved width is near-universal; quartered is aggressive and starts giving back acceptance on some configurations.
- Measured speedups are **end-to-end**; the cheaper draft forward only wins when it meaningfully reduces wall-clock. Low-batch, long-decode workloads benefit most.
- A narrower drafter is also easier to quantize aggressively — compounding wins with INT4 / FP8 drafters.
- Target hidden-state-attached drafters (EAGLE) are a different design point; NarrowSpec's claim is strongest for standalone-drafter-LM setups.

## Sources

- Paper: *NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding* — 2026 — [arXiv:2611.23835](https://arxiv.org/abs/2611.23835).
- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2022 — the SD acceptance rule.
