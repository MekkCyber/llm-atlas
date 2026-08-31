# Test-time scaling
*Taxonomy — methods that spend more inference compute to raise output quality.*

**TL;DR:** Once a model is trained, extra compute at inference can still buy accuracy. The methods split by *where the extra compute goes*: sampling more, verifying more, refining more, updating the model on-the-fly, or steering with cheap context tricks. Each has different scaling curves and different infrastructure requirements.

**Related taxonomies:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/_rewards.md](../post-training/_rewards.md)
**Depth files covered here:** [criticl](criticl.md) · [ttpo](../post-training/reasoning/ttpo.md)

---

## The problem

A fixed-weights model has a fixed best-of-one accuracy on a task distribution. But some tasks admit verification cheaper than solution (math checks, code tests, self-consistency across samples). When that holds, extra inference compute buys accuracy that no amount of training would have given cheaply. The question is *how* to spend the compute.

## The shared pattern

Every variant answers three questions:

- **Where does the compute go?** More samples, more verifier calls, more refinement steps, in-place model updates, or richer context.
- **What is the verifier?** A ground-truth check, a learned reward model, a self-consistency signal, or nothing at all.
- **Does the model change?** Weights frozen (context tricks only), or transiently updated per prompt.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Best-of-N | Sample N, pick best by scorer / verifier. | Cost scales with N; needs a verifier. | Verifiable tasks (math, code). |
| Self-consistency | Sample N, majority-vote the answer. | Weak signal for open-ended tasks. | Multiple-choice / short-answer reasoning. |
| MCTS-style search (see [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md)) | Tree search over reasoning steps with a value / process reward. | Requires PRM; heavy infrastructure. | Long-CoT reasoning with intermediate signal. |
| Process-reward reranking (see [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)) | PRM scores per step; pick trajectory with best step-wise reward. | PRM training / calibration cost. | When PRM is available. |
| [CritICL](criticl.md) | Prepend failure-mode critiques from a smaller sibling model. | Bounded by small-model failure diversity. | Same-family reasoning; low latency. |
| [TTPO](../post-training/reasoning/ttpo.md) | Update policy per prompt using self-generated signal. | Requires adapter reset; per-prompt overhead. | Hard prompts where sampling alone plateaus. |

## How to choose

- **Verifier available and cheap?** Best-of-N is the strong baseline. Increase N until returns flatten.
- **No verifier, multiple-choice / short-answer?** Self-consistency; simple and cheap.
- **Long-CoT reasoning with a PRM?** MCTS or PRM reranking — search dominates on hard problems.
- **Same-family small model available and latency-sensitive?** CritICL — one pass, no repeated sampling.
- **Hard prompts where none of the above budge?** TTPO — spend compute updating the model, not just sampling it.

Common composition: TTPO on a small compute budget, then best-of-N with PRM reranking of TTPO outputs. Search and update are orthogonal.

## Adjacent but distinct

- **Speculative decoding** — inference-time *speed* trick, not accuracy. Same latency budget, more tokens per second.
- **Long-CoT training** ([../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)) — pays the cost at *training* to shift the inference-time compute frontier.
- **Prompt engineering / few-shot** — technically free inference-time steering; taxonomy assumes at least one extra forward pass.

## Sources

- *CritICL* — Wu et al., 2026 — [arXiv:2608.27455](https://arxiv.org/abs/2608.27455)
- *TTPO* — Wang et al., 2026 — [arXiv:2608.27448](https://arxiv.org/abs/2608.27448)
- *Scaling Test-Time Compute Optimally* — Snell et al., 2024 — [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
