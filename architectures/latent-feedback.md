# Latent Feedback
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **vertical feedback channel** between decoding steps of an autoregressive transformer. At each step, the previous **top-layer hidden state** is fused with the sampled token embedding via a gated linear unit and fed back as the next input. Standard transformer, KV cache, and LM objective are all preserved; the model just gains a wider channel for non-verbalized computation to carry across positions. Introduced by the Full-bandwidth Transformer (Microsoft Research, 2026).

**Prereqs:** [transformer-block.md](transformer-block.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md), [multi-head-attention.md](multi-head-attention.md)

---

## What it is

An autoregressive transformer computes along two axes: **horizontally** across generated tokens (via attention) and **vertically** through model depth (via layer stacking). Between two decoding steps, the horizontal channel is wide — attention gives each token broad access to the past — but the vertical channel is narrow: only the sampled token identity re-enters the stack at the bottom. Everything the top layers computed is discarded.

Latent feedback widens that channel: keep the previous step's top-layer hidden state and merge it with the new step's input-token embedding, so non-verbalized computation can re-enter the stack with a fresh depth budget.

## How it works

At decoding step $t$, let $h_L^{(t-1)}$ be the previous step's top-layer hidden state and $e^{(t)}$ be the embedding of the just-sampled token $x^{(t)}$.

The next input to the stack is:

$$
z^{(t)} = \mathrm{GLU}\!\left([\, e^{(t)};\, h_L^{(t-1)} \,]\right)
$$

A gated linear unit fuses the two, controlling how much of $h_L^{(t-1)}$ leaks into the input vs. how much of $e^{(t)}$ dominates. The rest of the transformer stack — attention, FFNs, KV cache, LM head, tokenization — is unchanged.

**Training with parallel teacher forcing** normally would require the same computation to be available at every position, which the recurrence breaks. The paper introduces a **scheduled multi-pass objective**: latent feedback is introduced only late in pretraining, and a small fraction of deeper feedback passes are mixed in for stability. This preserves the ability to teacher-force long sequences in parallel while still training the feedback path.

## Why it matters

- **Rare free axis.** KV cache, LM objective, and decode cost per token are all preserved. Almost every "add capacity" trick sacrifices at least one.
- **Equivalent to more data.** 1B-parameter full-bandwidth transformers trained to 400B tokens match/approach standard transformers trained with **~1.5× more tokens** — a substantial effective-compute win.
- **Shorter reasoning traces at equal accuracy.** Because non-verbalized computation now persists across tokens, the model can defer some of the "thinking" to the latent channel instead of emitting it as visible CoT tokens. Cheaper inference for reasoning tasks.
- **Improves broad benchmarks.** Validation loss, 5-shot LM eval, math and coding generation, and instruction-tuned scores all move together — evidence the effect isn't a narrow benchmark artifact.

## Gotchas & tricks

- **Scheduled introduction is load-bearing.** Introducing latent feedback from step 0 destabilizes training — the recurrence entangles gradients across positions in a way that hurts convergence. Feed it in late once the model is well-conditioned.
- **Deeper-feedback pass mixing.** A small fraction of training batches use *deeper* feedback (e.g. feedback of $h_{L}$ two steps back) as a regularizer — without it, the model overfits to single-step feedback and struggles on long generations.
- **KV cache invariance.** Because the fusion happens *before* the first layer, the KV cache the attention layers write is unchanged in shape. No serving kernel changes required.
- **Not a substitute for depth or width.** The gain is on top of a well-chosen depth/width; you can't drop layers because you added feedback.
- **Scaling untested.** Reported at 1B / 400B tokens. Whether the 1.5× effective-tokens win holds at 100B+ is open.

## Sources

- Paper: *Full-bandwidth Transformer* — Wang et al., Microsoft Research, 2026 — [arXiv:2608.08888](https://arxiv.org/abs/2608.08888).
- Related: [mtp.md](../pre-training/mtp.md) — multi-token prediction is a different way to widen the vertical channel (extra prediction heads, not feedback).
