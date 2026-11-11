# Modality-Aware Expert Pruning (MAEP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **training-free** expert-pruning method for MoE-based multimodal LLMs that scores expert importance **separately for text and vision tokens** rather than averaging across modalities. Uses Δ-H Output, a direct "what breaks if I remove this expert" measure, and treats low-attention vision tokens as a stable pruning signal. Pruning ~50 % of experts cuts BF16 expert footprint by ~27 GB, outperforming modality-agnostic baselines on most multimodal benchmarks.

**Prereqs:** [./_moe.md](./_moe.md), [./aux-loss-free-balancing.md](./aux-loss-free-balancing.md)
**Related:** [./deepseek-moe.md](./deepseek-moe.md), [../multimodal/README.md](../multimodal/README.md), [../quantization/README.md](../quantization/README.md)

---

## What it is

In an MoE multimodal LLM, expert parameters are >90 % of model size. For deployment to tighter-memory tiers, expert pruning is the first lever. Prior work scores each expert's importance by aggregating some signal (routing frequency, output entropy, gradient magnitude) across *all* tokens it sees — which mixes text tokens and vision tokens into one score.

The problem: vision tokens activate experts very differently from text tokens. An expert that is noise for text may be load-bearing for vision; a modality-agnostic score prunes it, breaking the model on image inputs.

MAEP keeps per-modality tallies, scores each expert's importance *for text* and *for vision* separately, and uses a modality-balanced selection rule.

## How it works

1. **Expert scoring — Δ-H Output.** For each expert $e$, compute
$$
\Delta H(e) = \mathbb{E}_{t \in \text{tokens}}\left[\,\|y_t - y_t^{(-e)}\|\,\right]
$$
where $y_t$ is the MoE layer output and $y_t^{(-e)}$ is the output computed with expert $e$ removed. This directly measures the removal's impact on output.
2. **Modality split.** Maintain $\Delta H_\text{text}(e)$ and $\Delta H_\text{vision}(e)$ separately over a small calibration set.
3. **Vision-token refinement.** Vision tokens with **low attention** to the current query are more stable signals of baseline expert usage (they aren't being over-weighted by current context); MAEP weights the vision tally toward these.
4. **Prune.** Keep the top-$K$ experts by a modality-balanced score (e.g. max or weighted sum of the two per-modality scores), so no modality is starved.
5. **Serve.** The pruned model serves normally; no retraining, no fine-tune.

## Why it matters

- **Training-free** — zero fine-tune budget, drops into any served MoE MLLM.
- **~27 GB saved** in BF16 expert memory at 50 % prune rate, with better benchmark retention than modality-agnostic baselines.
- **Δ-H Output is reusable** — the "measure the removal effect directly" score beats proxy signals (entropy, routing frequency) in the paper's ablations and generalizes beyond MLLMs to any MoE where modality / domain subsets exist.
- Complements quantization — pruning reduces weight count, quantization reduces bytes-per-weight; both can stack on expert tables.

## Gotchas & tricks

- **Calibration set must be modality-balanced.** If text tokens dominate calibration, the vision score is noisy and the method's advantage over the baseline shrinks.
- **"Low-attention vision tokens" needs an attention backend that exposes per-token scores** — not all serving stacks surface this cleanly.
- **Δ-H computation cost.** Computing $\Delta H(e)$ for each expert costs one forward per expert on the calibration set. For $N = 256$ experts, this is 256× calibration-set forward passes — doable but not trivial. Amortize by subsampling tokens.
- **Doesn't help dense layers.** All compression stays in the MoE FFN; attention, norms, and embeddings are untouched. If those dominate memory in your model, pruning experts has diminishing returns.
- **Interacts with aux-loss-free balancing.** Pruning changes the empirical routing distribution; the aux-loss-free bias adapts automatically at serve time if left active, but freezing the bias at the pre-prune values is safer.

## Sources

- Paper: *Modality-Aware Expert Pruning for MoE-Based Multimodal Large Language Models* — Nota Inc. et al., 2026 — [arXiv:2611.05927](https://arxiv.org/abs/2611.05927)
