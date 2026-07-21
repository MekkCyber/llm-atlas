# On-Policy Delta Distillation (OPD²)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** **OPD²** replaces the usual on-policy distillation signal (KL to teacher) with the **delta signal**: the *difference* between a reasoning-tuned teacher and its own pre-tuning base. The delta captures what reasoning tuning *added*, filtering out the base capabilities the student already has, so the student spends its updates on residual reasoning skill rather than re-learning shared behavior.

**Prereqs:** [on-policy-distillation.md](on-policy-distillation.md), [_post-training.md](_post-training.md)
**Related:** [rejection-sampling.md](rejection-sampling.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [rlvr.md](rlvr.md)

---

## What it is

Given three models on the same tokenizer:

- $\pi_T$ — the **reasoning-tuned teacher** (e.g. an R1-style checkpoint)
- $\pi_B$ — the **base** the teacher was tuned from (e.g. the pre-RL checkpoint of the same model)
- $\pi_\theta$ — the **student** being trained

OPD² supervises the student toward the *delta* $\pi_T - \pi_B$ (in log space, this is a difference of logits) rather than toward $\pi_T$ directly. The intuition: $\pi_B$ already knows how to write English, follow format, and do easy arithmetic; those skills carry no reasoning-transfer signal. What's *new* about $\pi_T$ is where the reasoning capability lives.

## How it works

1. **Student rollout.** Student generates $o \sim \pi_\theta(\,\cdot\, \mid q)$ on prompt $q$ (on-policy — same setup as [on-policy-distillation.md](on-policy-distillation.md)).
2. **Teacher and base forwards.** For every prefix $(q, o_{<t})$, run $\pi_T$ and $\pi_B$ to get their next-token distributions.
3. **Delta target.** Form the delta signal, typically in log-space:
   $$\ell_T^{\text{delta}}(v \mid q, o_{<t}) \propto \log \pi_T(v \mid \ldots) - \log \pi_B(v \mid \ldots)$$
   Normalize to a distribution (softmax over vocab).
4. **Loss.** KL between student and the delta target:
   $$L_{\text{OPD}^2} = \sum_t \mathrm{KL}\big(\text{softmax}(\ell_T^{\text{delta}}) \,\|\, \pi_\theta(\,\cdot\, \mid q, o_{<t})\big)$$
5. **Optimize student only.** Teacher and base are frozen; two extra forward passes per step.

Empirically OPD² beats plain on-policy distillation on math, science, and code reasoning benchmarks — a short post-training run is enough to lift a student from base to reasoning-competent.

## Why it matters

- **Cleaner reasoning transfer.** Plain distillation wastes gradient budget on tokens the student already handles well. The delta target concentrates supervision on the residual — precisely the tokens where reasoning tuning changed the teacher.
- **Compatible with the R1-family teacher pattern.** Any RL-post-trained teacher paired with its own pre-RL base fits — DeepSeek-R1 / DeepSeek-V3-Base, Qwen-Reasoning / Qwen-Base, etc.
- **Drop-in for on-policy pipelines.** The change is one extra teacher forward and one subtraction; everything else stays the same.
- **Alternative to RL when a reasoning teacher exists.** For teams without RL infrastructure but with access to a reasoning-tuned teacher, OPD² is a cheap path to comparable capability.

## Gotchas & tricks

- **Requires a matched base.** You need $\pi_B$ (the exact pre-tuning checkpoint), not just any base model — architectures and tokenizer must match the teacher.
- **Delta can go negative in log-space.** Softmax over $\log \pi_T - \log \pi_B$ handles negative values naturally, but numerically stable implementations subtract the max first.
- **Base-vs-teacher must actually differ.** If $\pi_T \approx \pi_B$ on a subset of tokens (unchanged by tuning), delta is uninformative there. Filter to high-delta positions or accept the noise.
- **Temperature affects transfer.** A slightly higher-temperature delta target smooths over teacher idiosyncrasies; too high and the signal blurs.
- **Doesn't replace RL for capability discovery.** OPD² transfers *existing* reasoning capability. It can't find capability the teacher doesn't have — that's still RL's job.

## Sources

- Paper: *On-Policy Delta Distillation* — NAVER AI, 2026 — [arXiv:2607.15161](https://arxiv.org/abs/2607.15161).
- Related: *Distilling Reasoning Capabilities into Smaller Language Models* — the general R1-distillation pattern.
- Code: `github.com/naver-ai/opd2` (per the paper).
