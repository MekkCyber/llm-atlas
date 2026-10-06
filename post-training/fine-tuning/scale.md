# SCALE — Entropy-Gated Modulation of a Frozen SFT Delta
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Token-reweighting methods for SFT assign *nonnegative* coefficients to each token, so they can suppress or amplify an SFT update but cannot *reverse* a harmful learned feature once it's in the weights. SCALE (Selective Control of Adaptation via Local Entropy) freezes both the pretrained model and the SFT delta, then learns bounded token- and module-specific gates — including *negative* values — by minimizing predictive entropy alone. Beats the strongest token-reweighting baseline on math (37.84 / 43.60 / 36.57 averages across Qwen2.5-Math-1.5B / 7B and Qwen3-4B-Base) and leads on average code generation. Introduced in *Rethinking Token Reweighting for SFT* (2026, arXiv:2609.33463).

**Prereqs:** [README.md](README.md), [../_post-training.md](../_post-training.md)
**Related:** [../_rl.md](../_rl.md)

---

## What it is

A post-hoc correction layer applied *after* SFT. The pretrained weights $W_0$ and the SFT delta $\Delta W_{\text{SFT}}$ are both frozen; a bounded gate $g$ modulates how $\Delta W_{\text{SFT}}$ is applied. The gate can take negative values, so it can send any specific feature to *reverse* or *extrapolate* — a direction nonnegative weights can't reach.

The reframe: effective SFT correction can come from *controlling how already-learned residuals are used*, not from re-running SFT with different weights.

## How it works

Freeze $W_0$ and $\Delta W_{\text{SFT}}$. For token $t$ and module $m$, apply:
$$
W_{\text{effective}}(t, m) = W_0(m) + g(t, m) \cdot \Delta W_{\text{SFT}}(m)
$$
with $g(t, m) \in [-\alpha, \beta]$ for small bounds $\alpha, \beta$. The gate is a cheap learned function of the hidden state at position $t$ and the module $m$.

Train $g$ alone (everything else frozen) by minimizing **predictive entropy** on an unlabeled holdout:
$$
\mathcal{L}_{\text{SCALE}} = \mathbb{E}_t \big[\, H(\pi_{\text{effective}}(\cdot \mid x_{<t})) \,\big]
$$

Interpretation of a trained gate value:
- $g \approx 0$: suppress the SFT feature at this (token, module) — ignore what SFT learned here.
- $g \approx 1$: use the SFT feature unchanged.
- $g > 1$: extrapolate — use *more than* the SFT learned.
- $g < 0$: reverse — go the opposite direction from what SFT learned.

Nonnegative token reweighting can only reach the $[0, +\infty)$ half of this space, so it cannot undo a harmful feature after the fact.

## Why it matters

Argues that the SFT correction problem is better posed as "modulate the trained delta" than "retrain with better token weights." Three practical consequences:

1. **Cheap.** No full SFT rerun; the gate parameters are tiny relative to $W_0$ or $\Delta W_{\text{SFT}}$.
2. **Composable.** Any SFT run can be followed by a SCALE pass.
3. **Reversible.** Dropping the gate recovers vanilla SFT.

Results: math-reasoning averages 37.84 / 43.60 / 36.57 across Qwen2.5-Math-1.5B / 7B and Qwen3-4B-Base — above the strongest token-reweighting baselines — while staying competitive on general-retention benchmarks and beating all baselines on average code generation (HumanEval, HumanEval+, MBPP).

## Gotchas & tricks

- **Entropy alone can collapse the gate.** The bounded range on $g$ is load-bearing — unbounded gates minimize entropy by extrapolating indefinitely.
- **Needs unlabeled data that resembles deployment.** Entropy is a proxy; if the entropy-minimizing direction isn't aligned with task quality, gains disappear.
- **Not a replacement for RL.** SCALE corrects *within the SFT delta's span*. It can't learn capabilities absent from the SFT direction.
- **Module granularity matters.** Per-token-per-module gates work; per-layer-only gates are too coarse and lose the fine-grained correction wins.

## Sources

- Paper: *Rethinking Token Reweighting for SFT: Suppress, Reverse, and Extrapolate Learned Features* — Li, He, Gao et al., 2026 — [arXiv:2609.33463](https://arxiv.org/abs/2609.33463).
