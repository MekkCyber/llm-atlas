# Superposition Linearity in Transformers
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** If you feed a Transformer the **linear combination** of two prompts' hidden states, its next-token distribution is approximately the **average** of the two individual distributions. This "Superposition Linearity Hypothesis" is an *architectural* property that's strongest at initialisation, decays during pretraining, and can be restored with a light fine-tune — enough to decode two coherent streams from a single forward pass.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [README.md](README.md), [sae.md](sae.md)

---

## What it is

A claim, backed by measurements, that the Transformer's mapping from hidden states to output distributions is *nearly linear across inputs*. Formally, for two prompts $x_1, x_2$ with hidden states $h_1, h_2$ at some layer, and any mixing weight $\alpha \in [0,1]$:

$$p_\theta(\text{next} \mid \alpha h_1 + (1-\alpha) h_2) \approx \alpha\, p_\theta(\text{next} \mid h_1) + (1-\alpha)\, p_\theta(\text{next} \mid h_2)$$

The gap between LHS and RHS (measured in KL) is the paper's **linearity error**. It is small at random-init, grows during pretraining, and shrinks again after a short fine-tune.

## How it works

To *use* linearity, the authors introduce a **guided decoding** procedure:

1. Encode both prompts, mix their embeddings once (`αh_1 + (1-α)h_2`).
2. At each generation step, compute the joint logits from the mixed context, then **project** them onto each source's individual distribution to recover per-stream logits.
3. Sample two next tokens — one per stream — and feed each back into its own context for the next step; only the mixed hidden state is re-computed.

The net result is two coherent continuations per forward pass, at roughly the cost of one.

To *measure* linearity during training, the authors track the KL between the true distribution and the interpolated one on held-out prompt pairs. This curve is monotonically increasing with pretraining tokens, then drops sharply during 100–1k-step fine-tuning on paired-prompt data.

## Why it matters

- **Architectural, not emergent.** Linearity is highest at random initialisation, meaning it's a property of the Transformer's function class — an axis for theoretical analysis of attention, not just an empirical curiosity of trained models.
- **Cheap 2× throughput hint.** For workloads that can batch semantically distinct prompts, guided decoding is a much lighter parallelism trick than KV-sharing or continuous batching, and independent of the serving stack.
- **Training-dynamics signal for interp.** Linearity error is a scalar that changes during pretraining — a rare metric that isn't loss and correlates with something interpretable (how much the model "mixes" independent contexts).

## Gotchas & tricks

- Quality of the two streams **degrades** on prompts that require the model to disambiguate between the two contexts (e.g. one asks a question about the other). Guided decoding is best when the streams are independent.
- Fine-tune data matters: pairs of unrelated prompts drawn from the same domain restore linearity best; adversarially similar pairs don't help.
- The result is measured on residual-stream states; MLP outputs are more non-linear and linearity error is larger there.

## Sources

- Paper: *Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs* — Tikhonov et al., 2026, [arXiv 2609.29845](https://arxiv.org/abs/2609.29845) — the Superposition Linearity Hypothesis, its training dynamics, and guided decoding.
- Related: *Toy Models of Superposition* — Elhage et al., Anthropic, 2022 — the older sense of "superposition" (features packed into a low-dim space); this paper's usage is distinct but shares the linear-features intuition.
