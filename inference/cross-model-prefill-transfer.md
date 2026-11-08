# Cross-model prefill-to-autoregressive transfer (PACT)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Decouple prefill from decode across two different models: a **large sender** processes the prompt (prefill only, no generation); a **small receiver** does the autoregressive decode. Introduced by PLatour / PACT (2026). The same paper introduces **PACT-Check**, an ablation protocol that reveals a large fraction of observed gains come from *non-communication artefacts* rather than genuine cross-model transfer.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** none

---

## What it is

Prefill/decode disaggregation (vLLM, SGLang, Mooncake) runs prefill on dense compute and decode on bandwidth-optimised nodes — but both halves use the *same model*. PACT asks: what if the two halves use *different* models? Intuitively the prompt-processing stage is the compute-heavy one and benefits from a larger model; decode is the latency-sensitive one and benefits from a small one. If the small decoder can usefully consume what the big prefiller produced, you get a cheaper and faster system.

The paper's open framework is **PLatour**.

## How it works

PLatour extracts **transfer signals** from the sender's prefill (choices include last-layer KV, hidden-state summaries, or distilled soft-token prompts) and feeds them into the receiver as a prefix or additional attention keys/values. The receiver is fine-tuned to consume these signals.

Three variants from the paper:
1. **KV transfer.** Sender's final-layer KV cache is projected into the receiver's key/value space.
2. **Hidden-state prefix.** A summarisation head produces k tokens that the receiver prepends.
3. **Soft-prompt distillation.** Sender teaches the receiver via a short soft-prompt per prompt-family.

## The PACT-Check protocol

The paper's most important contribution is a diagnostic battery that measures how much gain is *actually* from sender-to-receiver transfer vs confounders:
- **Uninformative-sender baseline.** Replace sender output with random vectors — if the receiver still improves, the gain was from receiver training, not transfer.
- **Self-communication baseline.** Sender == receiver — if gains persist, transfer isn't cross-model.
- **Receiver-only training.** Train only the receiver on the same data without any sender signal.

Across tested PACT recipes, PACT-Check reveals that substantial fractions of the headline gains come from these confounders. The authors are unusually explicit about this.

## Why it matters

Cross-model prefill is likely the next step in disaggregated inference (smaller decode models, specialised prefillers per domain). PACT-Check is a necessary discipline for the subfield — every future paper claiming "big-prefill helps small-decode" should ablate against uninformative/self baselines or risk reporting artefacts.

## Gotchas & tricks

- **Vocabulary mismatch** between sender and receiver is a hard problem; the paper sticks to same-tokenizer families.
- **Latency budget.** Sender prefill is only a win if the receiver's decode phase is long enough that the sender's extra cost is amortised.

## Sources

- Paper: On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer — Lu et al., 2026 — [arXiv:2611.06976](https://arxiv.org/abs/2611.06976)
