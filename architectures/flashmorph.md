# FlashMorph — hybrid attention layer selection
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Hybrid attention models keep full attention on a subset of layers and swap the rest for cheap linear attention (or SSM). *Which* layers you keep matters — a lot. Prior methods pick layers greedily or via a fixed pattern (every other, first half, etc.), treating layer importance as independent. **FlashMorph** casts hybrid-layer selection as a **budget-constrained subset-optimization problem** and solves it with a scalable global selector, beating heuristic per-layer scoring at matched full-attention budgets. Turns Transformer→hybrid conversion into an optimization step, not a hand-tuned choice.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [attention](../fundamentals/attention.md)
**Related:** [mla](mla.md)

---

## What it is

Hybrid attention models replace some fraction $\rho$ of a Transformer's attention layers with linear-attention or SSM layers. Linear attention has $O(N)$ compute and constant-size state; full attention has $O(N^2)$ compute and $O(N)$ KV cache. Keeping some full-attention layers preserves quality; replacing the rest gives long-context savings.

The question: **which** layers do you keep?

- Fixed patterns (every $k$-th layer) — cheap, but wastes budget.
- Per-layer heuristic scoring (attention entropy, ablation deltas) — layerwise, treats layers independently.
- **FlashMorph** — global subset optimization: given a budget of $K$ full-attention layers out of $L$, pick the subset that maximizes a fast surrogate for downstream quality.

---

## How it works

**Objective.** Choose subset $S \subseteq \{1, \ldots, L\}$ with $|S| = K$ to keep full attention on. Convert the remaining layers to linear attention. The surrogate objective is a fast quality estimator — often calibration loss on a small held-out set, or a task-specific proxy computed without full fine-tuning.

**Subset optimization.** Naïve enumeration is $\binom{L}{K}$ — intractable at modern depths. FlashMorph uses a scalable selector that jointly ranks layers under the budget constraint, exploiting the surrogate's structure (e.g. submodularity when it holds, or a fast solver otherwise). The paper's "Fast LAyer Selection for Hybrid MORPHing" is exactly this selector plus its surrogate.

**Conversion step.** Once $S$ is picked, replace non-$S$ layers with linear-attention layers (weight-initialized from the full-attention weights when possible) and lightly fine-tune to close any remaining gap.

---

## Why it matters

- **Reveals layer-selection as a first-order lever.** Under matched full-attention budgets, choosing $S$ well is often worth more than choosing a fancier linear-attention variant.
- **Orthogonal to the linear-attention family.** Works with any drop-in cheap-attention layer — Mamba, sliding-window, linear attention, hybrid MLA, etc.
- **Cheap conversion, not retraining.** Because the surrogate is fast and the fine-tune is short, FlashMorph is deployable as a *conversion* step for existing pretrained Transformers.
- **Scales.** The selector runs at modern model depths without brute-forcing subsets.

---

## Gotchas & tricks

- **Surrogate choice is load-bearing.** A bad surrogate ($=$ misaligned with downstream quality) makes the "optimal" $S$ meaningless. Pick a surrogate whose ranking correlates with the workload you serve.
- **Non-uniform gains across layers.** Early layers and specific mid-layers are usually the ones that shouldn't be swapped. FlashMorph's picks are compatible with — but not equivalent to — this folk wisdom; run the optimization rather than assuming.
- **Post-conversion fine-tune is important.** Zero-shot conversion after picking $S$ leaves quality on the table. The paper's fine-tune is short but consistently helps.
- **Interaction with position encodings.** Some linear-attention variants require re-choosing positional encodings (e.g. dropping RoPE for a relative-bias). Selection should be joint if you're changing both.
- **Budget $K$ is a hyperparameter.** FlashMorph selects the best $S$ *given* $K$. Choose $K$ from the compute / quality Pareto curve.

---

## Sources

- Paper: *Morphing into Hybrid Attention Models* — Fudan / ByteDance Seed / CUHK, 2026 — [arXiv:2606.30562](https://arxiv.org/abs/2606.30562).
- Related linear-attention primitives referenced in the paper: Mamba / SSM layers, sliding-window attention, linear attention.
