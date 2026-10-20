# On-Policy Mechanism Distillation (OPMD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard on-policy distillation only matches the teacher's **outputs** on the student's own samples. OPMD also matches the teacher's **intermediate representations** — but first adapts them so the student's remaining layers can actually consume them. A thin "student-compatible teacher" projects the teacher's activations into the student's layer-wise state space; the student is then trained to reproduce those states on its own rollouts. At inference, no teacher is needed.

**Prereqs:** _(fills a graph gap — distillation basics live informally in [_post-training](../_post-training.md))_
**Related:** [_post-training](../_post-training.md) · [grpo.md](../grpo.md) · [rejection-sampling.md](../rejection-sampling.md) · [deepseek-r1.md](../../case-studies/deepseek-r1.md)

---

## What it is

Three generations of LLM distillation:

1. **Offline output distillation.** SFT the student on teacher-generated completions. Cheap, but teacher and student see different states.
2. **On-policy output distillation.** Student samples its own completions; the teacher scores them (or provides target distributions on them). Fixes the distribution-shift problem but still only transfers outputs.
3. **On-policy mechanism distillation (OPMD).** Add a term that matches **internal states** between a *student-compatible* projection of the teacher and the student's own activations, both evaluated on the student's own samples.

OPMD is in the third bucket. The crucial move is making the teacher's internal states consumable by the student's downstream layers — otherwise matching raw activations fails across architectures with different widths, depth splits, or vocabularies.

## How it works

**Student-compatible teacher.** For each layer index $\ell$ the student has, a small adapter projects the teacher's activations (at a mapped teacher-layer index) into the student's layer-$\ell$ state space. Adapters are tiny (linear or 2-layer MLP), trained alongside OPMD.

**On-policy rollouts.** The student samples text from its current policy. Teacher and student both run on those rollouts.

**Dual loss.** For each student layer $\ell$,

$$
L = L_\text{output}(\pi_\text{teacher}, \pi_\text{student}) \;+\; \lambda \sum_\ell \| A_\ell^{\text{student}} - \phi_\ell(A_\ell^{\text{teacher}}) \|
$$

$L_\text{output}$ is the usual KL (or forward-KL) on next-token distributions; the second term is **state-level distillation** through the adapter $\phi_\ell$. Both losses are evaluated on the student's own text, keeping the gradient on-policy.

**Inference.** Adapters and teacher are discarded; the student runs standalone.

## Why it matters

- **Middle ground between output and full feature distillation.** You get mechanism transfer where the student can use it, without the pathologies of trying to match activations across incompatible shapes.
- **Cross-family distillation.** Teacher and student can come from different families (Qwen → Llama, dense → MoE), because the adapters absorb the architectural mismatch.
- **Multi-teacher.** The adapter-per-layer design supports several teachers simultaneously: each teacher provides its own set of adapters, and the state loss sums over them.
- **Reported wins.** Highest average accuracy among tested distillation baselines across three teacher-student settings; additional gains in multi-teacher and cross-family SFT.

## Gotchas & tricks

- **Layer mapping is a hyperparameter.** OPMD maps student layer $\ell$ to a chosen teacher layer index (often linear interpolation). Picking the wrong mapping starves mid layers of signal.
- **$\lambda$ controls stability.** Too large and the student fights its output objective; too small and mechanism transfer is lost. Reported sweet spot is a small fraction (output loss dominates).
- **Train the adapters first.** A short warmup pass fits the adapters at a frozen student before enabling joint OPMD training.
- **Not RL.** OPMD is still distillation: no sampled reward, no advantage. The "on-policy" refers to the student generating its own text, not to a policy-gradient update.

## Sources

- Paper: *OPMD: On-Policy Mechanism Distillation: Beyond Output Supervision* — Yu, Li, Ren et al., 2026 — [arXiv:2610.14022](https://arxiv.org/abs/2610.14022)
