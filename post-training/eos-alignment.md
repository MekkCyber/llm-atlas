# Semantic EOS Alignment
*Depth — a small fix for the on-policy-distillation loss that stops length inflation at the source.*

**TL;DR:** In on-policy distillation, base students and post-trained teachers often place stopping probability on *different* EOS tokens even when their declared stop sets are identical. The KL loss then suppresses the student's preferred termination without reliably transferring the teacher's, and student responses grow unboundedly. Treating functionally equivalent EOS tokens as one **semantic stop action** removes most mismatch-induced length inflation across Qwen3, Llama, and Gemma.

**Prereqs:** [on-policy-distillation](on-policy-distillation.md), [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [reasoning/length-penalty](reasoning/length-penalty.md), [reasoning/long2short](reasoning/long2short.md)

---

## What it is

A tokenizer can emit several tokens that all *mean* "stop generating": `<|eot_id|>`, `<|end_of_text|>`, `<|im_end|>`, `</s>`, plus model-family-specific chat markers. When a distillation stack pairs two models trained under different post-training regimes, the two may *concentrate stopping mass* on different tokens from that same equivalent set.

The declared stop set (what the decoder treats as end-of-generation) can look identical, hiding this distributional mismatch. Under token-level KL, the student is then pushed away from its own EOS choice without being pulled toward the teacher's.

Semantic EOS alignment declares an *equivalence class* of stop tokens and merges their probabilities before computing the KL.

---

## How it works

1. Enumerate the tokens that decode to functional end-of-generation for both teacher and student.
2. Group them into a **semantic stop set** $S = \{s_1, \dots, s_k\}$.
3. Before computing token-level KL, aggregate probabilities:

$$
P_T'(S | \cdot) = \sum_{s \in S} P_T(s | \cdot), \quad P_S'(S | \cdot) = \sum_{s \in S} P_S(s | \cdot)
$$

Replace the individual $s_i$ entries with the aggregate. The KL now treats the stop action as one class regardless of which token spells it.

The fix is a few lines in the loss code, no model change.

---

## Why it matters

Without the fix, response length under OPD balloons across Qwen3, Llama, and Gemma — sometimes exhausting the generation budget. With the fix, length inflation drops substantially in all three families. Because response length drives serving cost linearly, this is a cheap and universal serving-cost win for any team using OPD to compress a reasoning model.

Yang et al. (2026) also show, via stage-wise analysis on K2-Horizon, that termination-mismatch explains only part of the length dynamics: a distinct **late-stage length inflation** persists even after EOS alignment. So the fix is necessary but not sufficient; downstream length control (length penalty, IDAC) may still be needed.

---

## Gotchas & tricks

- **Enumerate carefully per family.** Miss one EOS token and the mismatch reopens. Grep the tokenizer for `end`, `eot`, `eos`, `im_end`.
- **Chat-template ends count too.** In multi-turn chat models, the "end of assistant turn" marker is functionally a stop under most decoders.
- **Doesn't fix late-stage inflation.** Yang et al. show a distinct late-training length increase that survives EOS alignment. Combine with reward-shaping length control ([reasoning/length-penalty](reasoning/length-penalty.md)).
- **Watch decoder stop set drift.** Some inference runtimes hard-code a stop set independent of the tokenizer's; check that the runtime and the loss agree on $S$.

---

## Sources

- Paper: *When EOS Tokens Disagree: Understanding Length Inflation in On-Policy Distillation* — Yang et al., 2026 — [arXiv:2609.20511](https://arxiv.org/abs/2609.20511).
