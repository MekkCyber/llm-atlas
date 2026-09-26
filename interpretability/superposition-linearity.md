# Superposition Linearity
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** LLMs approximately behave like *linear* systems at the output distribution level: if you linearly combine the input embeddings of two unrelated prompts, the next-token distribution the model produces is close to the *superposition* of the two individual next-token distributions. This "Superposition Linearity Hypothesis" is measurable, is highest at initialization, *decays with pretraining*, and can be partially restored by a short fine-tune — enabling a guided decoder to extract two coherent continuations from a single forward pass.

**Prereqs:** [attention](../fundamentals/attention.md), [transformer-block](../architectures/transformer-block.md)
**Related:** [interpretability README](README.md)

---

## What it is

A behavioral rather than mechanistic test of the classic "superposition" story: instead of asking whether individual features share directions in activation space, ask whether the *end-to-end* output distribution decomposes linearly when the input is a linear mix of two prompts. Concretely, for token distributions $P_A$, $P_B$ from prompts $A$, $B$ alone, run the model on the input $\alpha \cdot \mathrm{embed}(A) + (1-\alpha)\cdot \mathrm{embed}(B)$ and compare its next-token distribution $P_\text{mix}$ against $\alpha P_A + (1-\alpha) P_B$.

## How it works

- **Measurement.** Sample pairs of unrelated prompts, mix their embeddings at a fixed $\alpha$, run one forward pass, compare $P_\text{mix}$ against the linear reference via KL or TV distance. Aggregate across pairs and layers.
- **Training-time dynamics.** Linearity error is near-zero at initialization (as expected for a mostly-linear-in-embedding system), then grows monotonically during pretraining as non-linear MLPs specialize.
- **Restoration.** A short fine-tune with a term penalizing linearity error recovers most of the lost linearity without materially degrading downstream capability.
- **Guided decoder.** Once the model is superposition-linear, mix two prompts, run one forward pass, then apply a decoding rule that alternately samples from the $P_A$ / $P_B$ "components" of $P_\text{mix}$, producing two coherent parallel continuations.

## Why it matters

- **Concrete black-box link to the superposition literature.** Most superposition work is internal (features, directions, SAEs); this predicts and tests a behavior at the output.
- **Serving payoff.** One forward pass, two independent continuations ≈ 2× throughput on paired-prompt workloads.
- **Pretraining pushes *against* the architecture.** Suggests transformers start compositional and lose some of it — a lever for future capability / efficiency trade-offs.

## Gotchas & tricks

- Linearity is measured on token distributions, not on hidden states — hidden-state linearity is much weaker.
- The guided decoder needs a rule to attribute each sampled token to $A$ or $B$; simple argmax attribution loses coherence on the tail — the paper uses a learned attribution head.
- Restoration fine-tune trades a small drop in per-prompt quality for the parallel-decoding capability; not a free lunch.
- The property is *approximate*: $\alpha=0.5$ mixes are cleanest; extreme $\alpha$ collapses back toward the dominant prompt.
- Not the same as prompt concatenation — mixing embeddings avoids doubling the sequence length, so the throughput win is real even on models with quadratic attention.

## Sources

- Paper: *Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs* — Tikhonov et al., AIRI / Skoltech, 2026 — [arXiv:2609.29845](https://arxiv.org/abs/2609.29845).
- Prior art: *Toy Models of Superposition* — Elhage et al., Anthropic, 2022 — the feature-superposition framing this paper tests behaviorally.
