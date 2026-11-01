# WISD: Weakening-Induced On-Policy Self-Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Build a *weaker* copy of a reasoning model — via low-bit quantization or magnitude pruning — then train the original model to maximize the log-probability gap between itself and this weakened sibling on its own on-policy rollouts. No external teacher, no privileged labels. Reported gains: +4.0 avg accuracy under quantized-weakening, +5.5 under pruning-weakening, across math and coding benchmarks.

**Prereqs:** [../_rl.md](../_rl.md), [../grpo.md](../grpo.md)
**Related:** [long-cot-rl.md](long-cot-rl.md), [../../quantization/fp8.md](../../quantization/fp8.md), [../../case-studies/deepseek-r1.md](../../case-studies/deepseek-r1.md)

---

## What it is

Classical self-distillation needs a stronger teacher. For frontier reasoning models there isn't one. WISD inverts the setup: use a *weaker* version of the current model (same architecture, degraded capacity) as the "anti-teacher" and train the original to pull away from it on tokens it itself generated. The signal is the per-token log-prob gap; the loop is strictly on-policy, teacher-free, and label-free.

## How it works

Given current policy $\pi_\theta$, construct a weakened policy $\pi_{\text{weak}}$ by one of:

- **Quantization-weakening:** quantize $\theta$ to low bit precision (the paper uses sub-4-bit).
- **Pruning-weakening:** zero out the smallest-magnitude weights at a fixed sparsity.

Both $\pi_{\text{weak}}$ constructions are deterministic given $\theta$; they are recomputed at each outer step.

Sample a rollout $o \sim \pi_\theta(\cdot \mid q)$ for a prompt $q$. Compute the per-token *weakening advantage*:

$$
A_t = \log \pi_\theta(o_t\mid q,o_{<t}) - \log \pi_{\text{weak}}(o_t\mid q,o_{<t})
$$

Then take a policy-gradient step with this $A_t$:

$$
\nabla_\theta J = \mathbb{E}_{o\sim\pi_\theta}\!\left[\sum_t \nabla_\theta \log\pi_\theta(o_t\mid \ldots)\cdot A_t\right]
$$

Plug into the standard PPO/GRPO objective (ratio clipping + KL-to-ref) with $A_t$ in place of the group-relative advantage. The signal is "the tokens where you most outperform your degraded self are the tokens worth reinforcing."

Because $\pi_{\text{weak}}$ is a *function of $\pi_\theta$*, the signal shifts with the policy — on-policy throughout.

## Why it matters

Teacher-free RL for reasoning models has been dominated by verifiable rewards (RLVR) and KL-to-ref anchoring. WISD gives a third signal that works even without a verifier *and* without a teacher — exactly the regime of open-ended reasoning. It also doubles as mechanistic evidence that quantization and pruning gaps encode something recoverable about which weights the model genuinely relies on for reasoning.

## Gotchas & tricks

- **Recompute $\pi_{\text{weak}}$ every rollout, not every step.** Rebuilding the quantized/pruned copy at every optimizer step is expensive; the paper amortizes across rollout batches.
- **Choose weakening severity to keep the gap informative.** Too mild → signal is noise; too severe → the weak model is useless and $A_t$ is near-constant. The paper's sub-4-bit and moderate-sparsity settings are starting points.
- **Pruning signal > quantization signal in their experiments.** +5.5 vs. +4.0 average lift. Hypothesis: pruning removes capacity more selectively, exposing which weights are reasoning-critical.
- **Combines with RLVR.** Use WISD when no verifier is available; stack with RLVR advantage when one is.
- **Not the same as rank-based self-rewarding.** No LLM judge is involved; the signal is purely the model-vs-weakened-self likelihood ratio.

## Sources

- Paper: *Weaken to Strengthen: Weakening-Induced On-Policy Self-Distillation for Large Reasoning Models* — Cai, Han, et al. — MIT · NVIDIA, 2026 — [arXiv:2610.38681](https://arxiv.org/abs/2610.38681).
- Reference: *DeepSeek-R1* — the kind of reasoning base this is applied to.
- Reference: *DeepSeekMath (GRPO)* — the host RL objective.
