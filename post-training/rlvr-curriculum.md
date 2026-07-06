# RLVR curriculum (transfer-aware)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Multi-domain RLVR mixes math, code, science, etc. on a fixed or hand-tuned schedule. **TAC** (Transfer-Aware Curriculum, Yang et al., 2026) is a bandit that picks which domain to sample next by combining **local learnability** (per-domain advantage magnitude) with **cross-domain transferability** (gradient-alignment between the domain's GRPO step and other domains' gradients). Both signals are free byproducts of the GRPO step already being computed — <1% wall-clock overhead — and the transferability term is what keeps the curriculum from over-committing to the easiest domain.

**Prereqs:** [rlvr](rlvr.md), [grpo](grpo.md)
**Related:** [rl-prompt-curation](rl-prompt-curation.md), [_rl](_rl.md)

---

## What it is

RLVR pipelines increasingly train on suites of domains — MATH, code (LiveCodeBench), science QA, IF, etc. — where reasoning skills should transfer. The **curriculum** is the per-step domain-sampling distribution. Fixed proportional sampling ignores current model state; hand-designed schedules require experiment budgets and don't adapt.

Learnability-based bandits already exist: sample the domain where the policy is *currently improving fastest*. But TAC's authors show this can over-commit to a single dominant domain, especially with imbalanced mixtures.

TAC upgrades this by asking: **does a step on domain $d$ help the other domains?** If yes, prefer it even when its own learnability is not maximal.

---

## How it works

For a batch of GRPO rollouts, per domain $d$:

- $A_d$ = mean absolute advantage on $d$ this step (learnability signal, free from GRPO).
- $g_d$ = projected GRPO gradient on $d$'s rollouts (free from the same step).

The transferability of domain $d$ to the remaining domains:

$$
T_d = \frac{1}{|D| - 1} \sum_{d' \ne d} \cos\!\big(g_d, g_{d'}\big)
$$

TAC picks the next domain by a bandit reward proportional to $A_d + \lambda \cdot T_d$ with UCB-style exploration. Because $g_d$ is a scalar per parameter dimension, projected onto the shared parameter space, cosine similarity is well-defined without extra forward passes.

Overhead: **<1% wall-clock** — no extra forwards, just the cosines and the bandit bookkeeping.

---

## Why it matters

- **Free curriculum signal.** RLVR pipelines routinely leave the domain-sampling schedule as a hand-tuned knob. TAC extracts a principled signal from computations already done.
- **Robust to imbalanced mixtures.** Learnability-only bandits over-commit to the dominant domain (its advantages dominate). The transferability term regularizes toward domains that also benefit others.
- **Concrete gains.** On a 6-domain reasoning suite with Qwen3-1.7B and Llama3.2-3B, TAC beats proportional sampling, a hand-designed schedule, and a learnability-only bandit by up to **+2.8 macro-accuracy points (10% relative)**.
- **Ablations confirm the transferability term is load-bearing.** Removing it collapses TAC back to the learnability-only baseline's failure mode.

---

## Gotchas & tricks

- **Gradient projection is the whole trick.** You need the per-domain GRPO gradient to compute cosines. In vanilla implementations these are already computed per micro-batch — just don't average them together before the bandit update.
- **$\lambda$ tuning.** Balances learnability vs transferability. TAC's ablations report gentle sensitivity around the tuned value; too large and the curriculum ignores what the model can currently learn.
- **Scales with domain count.** $O(|D|^2)$ cosines per step. For $|D| \le 10$ this is negligible; for hundreds of domains you'd batch or subsample.
- **Not just for RLVR.** The signal (per-mixture gradient alignment) is applicable to any multi-domain policy-gradient training, including RLHF with mixed reward types.
- **No effect if all domains transfer equally.** If your suite is highly homogeneous (e.g. all math sub-topics), TAC's advantage over learnability-only shrinks.

---

## Sources

- Paper: *Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR* — Yang, Liu, He, Zhang, Schölkopf, Jin, 2026 — [arXiv:2606.25178](https://arxiv.org/abs/2606.25178).
- Baselines: proportional sampling, hand-designed schedules, learnability-only bandits (see paper §5).
