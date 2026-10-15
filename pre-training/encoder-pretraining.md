# Encoder Pretraining — Dual-Head Masking (UltraBERT)
*Depth — pushing bidirectional encoder pretraining past where ModernBERT-era recipes saturate, by splitting the LM head by masking ratio.*

**TL;DR:** Bidirectional encoders are quietly resurgent — for retrieval, moderation, and cheap classification — and their pretraining recipes have been getting LLM-style upgrades (long context, more data, distillation). **UltraBERT** identifies three findings that reshape the recipe: downstream performance **saturates past a few trillion tokens**, **low masking ratios favor classification while high ratios favor retrieval**, and **distillation-with-pruning helps classification at low mask ratios but hurts retrieval at high ones**. The fix is a backbone with **separate LM heads for low vs. high mask ratios**, distilled only on the low-ratio head.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [mid-training.md](./mid-training.md) · [../post-training/README.md](../post-training/README.md)

---

## What it is

A pretraining recipe for a bidirectional (encoder-only) transformer intended to serve both classification and retrieval downstream heads. The paper's claim is that these two downstream tasks want *different* pretraining regimes, so the backbone should expose both.

## How it works

**Two LM heads sharing a backbone.** At pretraining time, the model has:
- A **low-mask-ratio head** (e.g. mask 15% of tokens), optimized to produce rich token-level representations good for classification.
- A **high-mask-ratio head** (e.g. mask 40%+ of tokens), optimized under heavier masking to produce rich sequence-level representations good for retrieval.

Each training step randomly samples a masking ratio and routes through the corresponding head. The backbone sees both regimes; each head sees only its own.

**Distillation only on the low-ratio head.** Prior encoder recipes often use **distillation-with-pruning** (DWP) to compress during pretraining. UltraBERT finds DWP helps at low mask ratios (classification) but *hurts* at high mask ratios (retrieval). Fix: apply distillation only to the low-ratio head; keep the high-ratio head at full capacity.

**Saturation finding drives the data budget.** Pretraining past ~a few trillion tokens gives diminishing returns on downstream metrics. UltraBERT caps the pretraining budget accordingly and reinvests compute into the dual-head structure instead of more data.

## Why it matters

- Encoders are a serious deployment target (cheap classification, moderation, retrieval). UltraBERT reports **91.0 on GLUE** and **47.7 on BEIR** — SOTA for encoders at the tested scale.
- Rejects the "one objective rules them all" assumption that most encoder recipes inherit from BERT. Downstream head choice should feed back into pretraining objective choice.
- Clear practical rule: **classify → low mask, DWP; retrieve → high mask, no DWP**.

## Gotchas & tricks

- **Mask-ratio sampling schedule matters.** A simple 50/50 split works; biased schedules (more low-mask early, more high-mask late) can trade a few points. The paper reports the uniform split.
- **DWP on high-mask hurts in a specific way.** Retrieval loss doesn't go up during pretraining with DWP; downstream BEIR drops. The failure is only visible on the downstream eval, so monitor it.
- **Not every encoder improvement transfers.** LLM-era tricks (RoPE, FlashAttention) are already in modern encoders; UltraBERT's contribution is on *objective*, not architecture.
- **Scaling-law question.** The saturation finding is at the tested scale; whether it holds past the frontier is unclear.
- **Composability with task-adaptive mid-training.** The low/high split is set in pretraining; downstream recipe can still mid-train on domain data with its own masking choice.

## Sources

- Paper: *UltraBERT: Beyond the Limit of BERT Pre-Training* — Lee, Kang (and a third author) — Korea University / AIGEN Sciences / Amazon, 2026 — arXiv:2610.15028.
