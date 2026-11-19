# DSpark
*Depth — block-wise speculative decoding with a Markov dependency head on a short DFlash drafter.*

**TL;DR:** Xing4's inference-side successor to MTP. A 5-layer DFlash drafter proposes **7 positions in parallel** from the target's intermediate representations; a lightweight **Markov head** applies logit corrections between adjacent proposed tokens; a **confidence head** estimates per-position acceptance probability. Delivers 2.67–3.77× decode throughput over base and 1.48–2.10× over MTP3.

**Prereqs:** [../pre-training/mtp](../pre-training/mtp.md)
**Related:** [../architectures/multi-head-attention](../architectures/multi-head-attention.md)

---

## What it is

A **parallel + dependency-corrected drafter** for speculative decoding. DSpark sits in the same architectural slot as MTP-at-inference (which Xing4 originally kept for training only) and as DNeo-style parallel drafters, but it combines two ideas:

1. **Block-wise parallel proposal** — the drafter emits a block of 7 tokens in one pass.
2. **Markov dependency correction** — a small head applies logit corrections that condition each proposed position on the previous one, so the parallel draft isn't blind to adjacency effects.

The **confidence head** is the third piece: it estimates per-slot acceptance probability so the runtime can decide whether to even verify long blocks.

## How it works

```
target hidden states h_1..h_L  (from multiple verifier layers)
                 │
                 ▼
DFlash drafter (5 layers, conditioned on multi-layer hidden states)
                 │
                 ▼
      parallel: draft logits  z_1, z_2, …, z_7
                 │
                 ▼
Markov head applies:  z_j  ←  z_j + M(s_{j-1})     # dependency on prior slot
                 │
                 ▼
Confidence head:  c_j = σ(…)  predicting P(accept at slot j)
                 │
                 ▼
Runtime truncates the block at low c_j (optional)
                 │
                 ▼
target model verifies the block → accepts prefix
```

**DFlash backbone** is the Xing4 project's short drafter architecture — five transformer layers that read the target's hidden state at several layers (not just the last), matching the "multi-layer features matter" lesson that DNeo also encodes in Mixture-of-Features Attention.

**Markov head.** A small module that applies additive logit corrections based on the *previous proposed token*. It's not full autoregression (which would defeat parallelism) but a one-step dependency correction that catches adjacency-level mistakes (e.g., matching brackets, verb agreement).

**Confidence head.** Trained to predict acceptance from draft representations and previous-token info. Lets the serving stack cut verification short when confidence drops, rather than committing to a fixed block size.

## Why it matters

MTP-at-inference (used in DeepSeek-V3) hits ~85–90% acceptance and ~1.8× speedup. DSpark pushes this to **~3× end-to-end** by combining parallel proposal, dependency correction, and adaptive block size. It's the strongest single-system speculative-decoding result in the current wave of frontier open-weight reports, and it's the reference design for "drop MTP at inference, run a short dedicated drafter instead."

On a 285B Xing 4 model:
- **GSM8K:** 115.8 tok/s (3.27× base, 1.82× MTP3)
- **HumanEval:** 134.7 tok/s (3.77× base, 2.10× MTP3)

The HumanEval gap is the headline — code's higher adjacency regularity is where the Markov head earns its keep.

## Gotchas & tricks

- **Requires multi-layer hidden-state access from the target.** Same runtime constraint as DNeo — the target model must expose intermediate activations, not just the last layer.
- **Markov head is adjacency-only.** It corrects k→k+1 dependency, not longer-range coherence. For long-range coherence the drafter has to be good enough; the Markov head won't save it.
- **Confidence head must be calibrated.** An over-confident head commits to blocks that waste verification compute; an under-confident head truncates to length 1 and loses all parallelism. The paper doesn't report calibration curves; practitioners should.
- **Acceptance rate not reported.** All gains are reported as throughput × base/MTP3. The acceptance rate and mean accepted tokens per call are left implicit — something to measure yourself if you port the recipe.
- **DSpark replaces MTP at inference, not at training.** MTP is still trained as an auxiliary objective; DSpark loads a *separately trained* drafter at serving time.

## Sources

- Paper: *TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks* — TeleAI, 2026 — [arXiv:2611.22115](https://arxiv.org/abs/2611.22115)
- Related contemporaneous work: *DNeo: Your Parallel Draft Model Needs Better Target Feature Encoding* — Bergsma et al., 2026 — [arXiv:2611.21925](https://arxiv.org/abs/2611.21925). Shared "multi-layer features matter" principle; added in the sibling `kg-update/2026-11-18` PR as `inference/dneo.md`.
