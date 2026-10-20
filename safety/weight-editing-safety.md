# Weight-Editing for Safety
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-free way to make an instruction-tuned LM resist specific harmful behaviors. Score each weight by its **magnitude × graded activation contrast** between paired "do the bad thing" and "do the good thing" prompts, then **rescale the top 4% per weight matrix**. Because the change lives in weights, prompt manipulation can't undo it; because only a tiny fraction of weights move, general capability is preserved.

**Prereqs:** [refusal-suppression.md](refusal-suppression.md)
**Related:** [_attacks.md](_attacks.md) · [competing-objectives.md](competing-objectives.md)

---

## What it is

Three ways to make an instruction-tuned LM refuse a harmful behavior:

1. **RLHF / safety fine-tuning** — retrain the model. Robust, expensive, static.
2. **System-prompt safeguards** — add a rule to the prompt. Cheap, flexible, easily jailbroken.
3. **Activation steering** — add a direction to the residual stream at inference. Cheap, prompt-robust at a single layer, but brittle across prompt formats.

Weight editing is a fourth option. It modifies the **model weights** (not activations or prompts) based on a tiny calibration set, with no gradient steps and no training loop. The result is a model whose behavior change survives any prompt rewrite because the change is in the parameters themselves.

## How it works

**Calibration data.** For the target behavior (e.g., instructed lying), collect a small set of **paired prompts**:
- "Bad" variant: the prompt that elicits the undesired behavior.
- "Good" variant: a matched prompt (same semantic content) that elicits the desired behavior.

A few dozen pairs suffice.

**Weight scoring.** For every weight $w_{ij}$ in every linear layer:

$$
\text{score}(w_{ij}) = |w_{ij}| \cdot \big| \mathbb{E}_\text{bad}[a_i \cdot \mathbb{1}[w_{ij} \to a_j]] - \mathbb{E}_\text{good}[a_i \cdot \mathbb{1}[w_{ij} \to a_j]] \big|
$$

In plain English: multiply the weight's magnitude by the **difference in its contribution** between the bad and good forward passes. High-score weights are those that are both large and used differently across the two conditions.

**Top-k selection.** Rank weights by score within each matrix and keep the **top 4%**. The scoring rule is finer than set-difference ("fires here but not there") because it uses **graded** activation contrast rather than a hard mask.

**Rescaling.** Shrink the top-scoring weights toward zero (not delete — rescale, which preserves gradient flow if the model is later fine-tuned). The rescale factor is a hyperparameter; the paper reports a sweet spot that trades targeted-behavior reduction against capability loss on standard benchmarks.

## Why it matters

- **Prompt-robust.** Because the edit is in weights, adversarial prompt rewrites cannot undo it; system prompts and single-layer steering can be bypassed by format changes, this cannot.
- **Narrow, surgical.** 4% of weights move; general capability on standard benchmarks is preserved.
- **Composable.** Combining weight editing with a system-prompt safeguard gives a further reduction (the paper's sexism example: ~58% → ~24% from weights alone, down to ~12% when combined with a fairness prompt).

## Gotchas & tricks

- **Pair quality is load-bearing.** Mismatched pairs (different topic as well as different behavior) leak confounding signal into the score. Keep pairs semantically matched.
- **Per-matrix top-k, not global.** Scoring is normalized within each weight matrix so later-layer matrices (which have different magnitude distributions) don't dominate.
- **Not safety-training.** The method targets *specific instructed behaviors* — it won't generalize to behaviors you didn't pair-example. For broad safety training, you still want RLHF.
- **Reversible.** The rescale factor is a scalar; multiplying by its inverse restores the original weights. Useful for ablations; a liability if an attacker has write access to the weights.
- **Also a probe.** The method's success implies the targeted behavior is encoded sparsely enough to find with activation contrast on a few pairs — relevant for interpretability work.

## Sources

- Paper: *Resisting Harmful Instructions through Training-Free Weight Editing* — Yang, Schapiro, Borhani, Ortega, Zheng, Zhai, 2026 — [arXiv:2610.21189](https://arxiv.org/abs/2610.21189)
