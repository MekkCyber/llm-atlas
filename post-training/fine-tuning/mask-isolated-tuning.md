# Mask-Isolated Tuning
*Depth — one specific technique, grounded in its source paper.*

**TL;DR:** When adding a new modality (or capability) to a pretrained model on a **compact parameter budget**, standard fine-tuning trades acquisition for forgetting — the new signal overwrites parameters that carried old capability. Mask-isolated tuning scores each pretrained parameter's **significance** to the existing capability, freezes the top-significance **critical subspace**, and only updates the low-significance **dormant subspace**. Introduced by the Splash paper (Ewha, 2026) for tactile-modality alignment in MLLMs.

**Prereqs:** [README.md](./README.md)
**Related:** [../../multimodal/README.md](../../multimodal/README.md)

---

## What it is

A parameter-efficient fine-tuning strategy for **modality expansion under a fixed parameter budget**. The problem it addresses: for compact MLLMs, there aren't enough parameters to add a new sensory modality *and* preserve existing vision-language reasoning if training touches all weights. LoRA-family methods add capacity in a low-rank delta; mask-isolated tuning instead asks *which of the base model's existing weights are safe to overwrite* and confines updates there.

The Splash paper applies this to giving compact MLLMs a tactile sense (friction, compliance) without losing vision-language ability. The recipe generalizes.

## How it works

**Score parameter significance.** For each pretrained parameter $\theta_i$, compute a significance score against the frozen model's capability — typically an **empirical Fisher information** on a probe dataset:
$$s_i = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{probe}}} \left[ \left(\frac{\partial \log p_{\theta}(y \mid x)}{\partial \theta_i}\right)^2 \right]$$
High $s_i$ means small changes to $\theta_i$ significantly perturb the model's outputs — it *matters*.

**Partition into critical vs dormant.** Rank parameters by $s_i$. Take the top-$\alpha$ fraction (typical $\alpha = 0.5$–$0.7$) as the **critical subspace**; the rest is **dormant**. This produces a binary mask $\mathbf{m} \in \{0, 1\}^{|\theta|}$: 1 = frozen, 0 = trainable.

**Train only the dormant slice.** During fine-tuning on the new-modality data, gradients are masked:
$$\theta_{t+1} = \theta_t - \eta (1 - \mathbf{m}) \odot \nabla \mathcal{L}(\theta_t)$$
Critical weights never move; new-modality alignment gets injected into the dormant subspace only.

**No new parameters added.** Unlike LoRA, the tuned model has the same parameter count as the base — no adapter delta at inference time.

## Why it matters

- **Non-destructive modality expansion on compact budgets.** Preserves existing capability without adding parameters and without collecting a full replay set of old-modality training data.
- **Complements LoRA rather than competing.** LoRA answers "what low-rank delta to add"; mask-isolated tuning answers "which base weights to hold fixed." Both can be stacked.
- **General recipe.** The parameter-significance framing transfers beyond tactile: any continual-learning problem where a compact base model needs a new capability faces the same acquisition-vs-forgetting trade.
- **The Splash paper demonstrates it** yields tactile alignment in an MLLM without catastrophic loss of vision-language reasoning — the first application to modality expansion in an MLLM at compact scale.

## Gotchas & tricks

- **Probe-set sensitivity.** The Fisher score depends on the probe dataset. Too narrow → protects only a slice of capability; too broad → protects everything and the dormant subspace is starved.
- **$\alpha$ is a tradeoff.** Small dormant slice → strong preservation, weak acquisition. Large dormant slice → the opposite. Author uses $\alpha \approx 0.6$.
- **Fisher approximates locally.** A weight with low local Fisher may still be globally important for a rare capability the probe set didn't sample. Combine with a small replay set for safety.
- **Not equivalent to freezing layers.** Mask-isolated tuning selects *individual parameters* across all layers, not whole layers. Coarser layer-freezing is a related but weaker baseline.
- **Multi-round expansion.** Adding a second modality after a first requires re-scoring on the *union* of prior capabilities; naively re-using the original mask overwrites what the first expansion added.

## Sources

- Paper: *Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs* (Splash) — Yoon, Yu, Park, Lee (Ewha Womans University), 2026 — arXiv:2607.00302.
- Related lineage: Fisher-information continual learning (Kirkpatrick et al., "Elastic Weight Consolidation," 2017) — same significance-scoring idea, applied for regularization rather than binary masking.
