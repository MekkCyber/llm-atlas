# Speculative Decoding
*Taxonomy — accelerate autoregressive decoding by generating candidate tokens with a cheap drafter and verifying them in parallel with the target.*

**TL;DR:** Autoregressive decoding is latency-bound: one forward pass per token. Speculative decoding runs a small, cheap **drafter** ahead of the big **target** model, proposing $k$ tokens at once; the target verifies them in a single forward pass and accepts the prefix of tokens where its own distribution agrees. Lossless (acceptance mimics target sampling) with 1.5–3× end-to-end speedups typical. Variants differ in drafter architecture, training coupling, and how aggressively they share state with the target.

**Related taxonomies:** *(none yet in this folder)*
**Depth files covered here:** [narrow-spec](narrow-spec.md) · *(VSpark, EAGLE, Medusa, DFlash depth files to come as added)*

---

## The problem

For an $N$-parameter target model, each generated token costs one forward pass at batch size 1. Inference latency is pin-pointed by that loop; even at FP8 and highly optimized kernels, the round-trip per token is tens of ms at frontier scale. Techniques that reduce per-token latency without shrinking the model face a hard constraint: the target's own distribution must stay the sampling truth.

## The shared pattern

```
loop:
  draft = drafter.generate(prefix, k tokens)                # cheap, autoregressive
  logits = target.forward(prefix + draft)                   # one parallel pass
  accepted = []
  for i in 0..k-1:
      accept with prob min(1, target_p[i] / drafter_p[i])
      if accepted: accepted.append(draft[i])
      else: break
  bonus = sample from (target distribution at the first rejected position)
  prefix.extend(accepted + [bonus])
```

Every variant of speculative decoding instantiates two design choices:

1. **How the drafter is produced** (separate small model, trained head on top of the target, n-gram, multi-token-prediction head).
2. **How closely the drafter mirrors the target's hidden state** (independent; shares embeddings; shares final-layer features; shares everything except final projection).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Vanilla speculative decoding (Chen et al. 2023 / Leviathan 2023) — *no depth file yet* | Separate small draft model | Draft model must exist and be good | Any setting; baseline |
| Medusa — *no depth file yet* | Multiple decoding heads on the target predict the next $k$ tokens in parallel | Target weights grow; no shared cache with original | When a separate drafter is impractical |
| EAGLE / EAGLE-3 — *no depth file yet* | Draft model fed with the target's last-layer features | Needs feature passing between drafter and target | Highest acceptance rates reported |
| DFlash — *no depth file yet* | Draft shares vocab projection with the target; end-to-end tuned | Tighter coupling; retraining when target updates | Strong fit for a single fixed target |
| [narrow-spec](narrow-spec.md) | Draft's hidden width need not match target; vocab head can be folded into the target's | Narrower drafter has less interior capacity | Single-user, low-batch inference |
| VSpark — *no depth file yet* | Speculative decoding adapted to vision-language-action models: noise-calibrated rollouts, feature alignment, reuse of rejected drafts | VLA-specific | Autoregressive VLA policies with action tokenizers |
| N-gram / lookup drafters — *no depth file yet* | Draft from a prefix-indexed n-gram table of the corpus | Trivial to run; low acceptance on open domains | Code and repetitive text; cheap baseline |

## How to choose

**Default for a frontier LLM in production:** EAGLE-3. Highest acceptance rates; adds a separate small drafter that is tightly coupled to the target.

**Single-user chat / coding agent latency floor (batch size 1):** narrow-spec on top of an EAGLE/DFlash drafter. The drafter's hidden width is a cheap axis; narrowing drops a sizeable fraction of per-draft cost without hurting acceptance much.

**Vision-language-action policies:** VSpark. Vanilla spec decoding generalizes poorly to action-token distributions; the noise-calibrated training and rejected-draft reuse are the key adapters.

**If retraining the drafter per target revision is prohibitive:** Medusa heads or separate small models decoupled from the target's internal features.

## Adjacent but distinct

- **Multi-token prediction (MTP)** at pretraining time: teaches the target itself to predict multiple tokens per step. Different objective at training; some overlap at inference. See [../pre-training/mtp](../pre-training/mtp.md).
- **Parallel decoding / lookahead decoding:** draft-free parallel strategies; usually dominated by learned drafters at matching compute.
- **Continuous batching:** throughput optimization. Composes with speculative decoding but targets a different metric.

## Sources

- Paper: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023.
- Paper: *Accelerating Large Language Model Decoding with Speculative Sampling* — Chen et al., 2023.
- Paper: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — Cai et al., 2024.
- Paper: *EAGLE / EAGLE-3* — Li et al., 2024–2025.
- Paper: *NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding* — 2026 (NVIDIA).
- Paper: *VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models* — Yoo et al., 2026 (KAIST AI).
