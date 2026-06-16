# LoRA-α — Scaling Factor as a First-Class Optimization Knob
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The LoRA scaling factor $\alpha$ is often treated as a redundant proxy for the learning rate. It isn't. A signal-drift analysis shows LoRA's low-rank decomposition spectrally suppresses certain gradient directions; only $\alpha$ — applied to the $BA$ product before residual addition — offsets that suppression. The practical recipe: set $\alpha$ by a **sublinear-in-rank** rule (not the common $\alpha = 2r$ / $\alpha = r$ defaults), and use the *full fine-tuning* learning rate directly. Recovers most of the accuracy gap to full FT across tasks without LR re-tuning.

**Prereqs:** [README.md](README.md) (LoRA basics — depth file not yet written)
**Related:** [../_post-training.md](../_post-training.md)

---

## What it is

LoRA parameterizes a weight update as $\Delta W = (\alpha / r) BA$ with $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, rank $r$. The factor $\alpha / r$ is folk-rescaled to "stabilize LoRA at different ranks." Common practitioner guidance: pick $\alpha = 2r$ or $\alpha = r$, then sweep the learning rate.

The paper shows this folk rule is incorrect on two counts:

1. The optimal $\alpha$ grows **sublinearly** with rank, not linearly.
2. $\alpha$ is *not* equivalent to a learning-rate multiplier — it has a separate effect on gradient geometry that LR cannot reproduce.

The resulting recipe **LoRA-α** picks $\alpha$ from a sublinear rule (e.g. $\alpha \propto \sqrt{r}$) and uses the same LR you'd use for full FT.

---

## How it works

### Signal-Drift framework

Decompose the LoRA gradient through the SVD of the base weight $W = U \Sigma V^\top$. The low-rank update $BA$ projects gradients onto a small subspace; directions aligned with the top singular vectors of $W$ are *suppressed* by an amount that depends on rank but **not** on the learning rate.

- A higher learning rate scales the *entire* update uniformly — it can't recover suppressed directions, only inflate everything else (often hurting).
- A higher $\alpha$ scales the *projected* update before residual addition, which differentially affects suppressed directions because they re-enter the residual stream with the boost.

This asymmetry is the "signal drift": $\alpha$ moves the effective signal in a direction LR cannot.

### Sublinear scaling rule

The paper derives that the optimal $\alpha$ scales as $\alpha^* \propto r^{1/2}$ (sublinear), not $\alpha \propto r$ (linear, the common heuristic). Intuitively: doubling rank halves per-direction suppression, so the compensation $\alpha$ needs is square-root, not linear.

### Recipe

1. Choose rank $r$ from compute / memory budget (unchanged).
2. Set $\alpha = c \cdot \sqrt{r}$ for a small constant $c$ (paper reports tuned $c$ per task; works across tasks within a narrow range).
3. Use the **full-FT learning rate** for the model size, *not* a LoRA-adjusted one.
4. Train as usual.

---

## Why it matters

- **Reuses full-FT hyperparameter intuition.** Practitioners don't need to re-sweep LR for every (model, rank, task) combination.
- **Closes the accuracy gap.** Reported to recover most of the gap between LoRA and full FT across diverse benchmarks without architectural changes.
- **Cheap to adopt.** Two numeric changes ($\alpha$ rule + LR choice); no new code paths.
- **Implies prior LoRA results were under-tuned.** Many published LoRA-vs-full-FT comparisons may have been using a suboptimal $\alpha$, understating LoRA's true ceiling.

---

## Gotchas & tricks

- **Sublinear-vs-linear matters most at high ranks.** At $r = 8$ the two rules give similar $\alpha$; at $r = 128$ they differ substantially and the difference is empirically visible.
- **Don't apply LR warmup to $\alpha$.** $\alpha$ is fixed; only the LR schedule moves. Treating $\alpha$ as part of the LR schedule recreates the original confound.
- **DoRA / variants.** The signal-drift analysis is derived for vanilla LoRA. DoRA, AdaLoRA, and other variants change the parameterization and need their own derivation; the sublinear-$\alpha$ rule isn't guaranteed to transfer.
- **AdamW assumed.** The analysis uses AdamW's per-coordinate scaling. SGD-trained LoRA may behave differently — under-explored in the paper.
- **Per-layer $\alpha$ is an open question.** The paper sets a single $\alpha$ for all LoRA-adapted layers. Per-layer schedules might further close the gap but aren't studied.

---

## Sources

- Paper: *The Hidden Power of Scaling Factor in LoRA Optimization* — Zhang et al., JD · UCAS · NKU, 2026 — [arXiv 2606.12883](https://arxiv.org/abs/2606.12883).
- Background: *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., 2021 — the original LoRA parameterization with $\alpha$.
