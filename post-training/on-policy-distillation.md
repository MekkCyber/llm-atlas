# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a student LM on **its own responses**, with per-step targets taken from a fixed teacher model's distribution. Unlike offline KD (train student on teacher's samples) or SFT distillation (teacher labels fixed dataset), OPD rolls out from the student's current policy, so the distillation data is always drawn from where the student actually generates. It has become the default scalable recipe for R1-style reasoning distillation and for agent-policy training. Studied as a quantizable, scheduleable system in *How Much Training-Inference Mismatch Can OPD Tolerate?* (2026).

**Prereqs:** [./_post-training.md](./_post-training.md), [./grpo.md](./grpo.md)
**Related:** [./dpo.md](./dpo.md), [./mismatch-budget.md](./mismatch-budget.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Three KD-family recipes for LLMs, in decreasing order of staleness:

- **Offline KD** — teacher samples a fixed dataset; student trains on it. Cheap, but student never sees its own failure modes.
- **SFT distillation** — teacher labels a fixed-prompt dataset; student trains on labels. Still fixed-prompt.
- **On-policy distillation** — student samples its own responses from prompts; teacher scores each response token-by-token (either hard labels at $\arg\max$, or soft KL to the teacher distribution); student minimizes KL to teacher at every token.

OPD's advantage is coverage: the student is distilled exactly where it produces outputs, so systematic student errors are exactly what the teacher corrects.

## How it works

Per batch of prompts $Q$:

1. Student generates responses $o \sim \pi_\theta(\cdot \mid q)$, typically with a fast rollout engine (vLLM/SGLang) that may use quantization (FP8, INT4) or async weight sync.
2. For each token position $t$, teacher computes $\pi_T(\cdot \mid q, o_{<t})$ (full distribution or top-$k$).
3. Student loss at token $t$ is $\mathrm{KL}\!\left(\pi_T \,\|\, \pi_\theta\right)$ (forward KL, standard) or the reverse KL for mode-seeking behavior.
4. Student parameters updated; rollout engine periodically re-synced to student weights.

The loop is deceptively simple; the engineering is almost entirely about keeping the rollout engine (quantized, async, batched) in sync with the student (BF16/FP32, synchronous) while bounding the resulting *training-inference mismatch*.

## Why it matters

- Underlies modern reasoning distillation: R1 → smaller models, Kimi k1.5 → open distillations, and many agent-policy training recipes.
- Rollout cost dominates — multiple seconds per sample at long context — so rollout engines quantize and async-sync to recover throughput. This introduces the mismatch problem that [mismatch-budget](./mismatch-budget.md) formalizes.
- Empirically, OPD tolerates **more** training-inference mismatch than online RL before quality degrades — a useful asymmetry that justifies aggressive rollout optimization.

## Gotchas & tricks

- **Not the same as online RL.** No reward, no advantage — pure KL to a fixed teacher. Dynamics are closer to SFT than to PPO/GRPO.
- **Teacher needs to score, not generate.** Cheap-rollout tricks that destroy the teacher's output distribution (aggressive quantization) hurt here even though they would be fine for the student rollout.
- **Length matters.** Long reasoning traces make per-token KL noisy; many implementations use a length-normalized loss or only distill on the first $L$ tokens.
- **FP8 rollouts generally safe**, per the mismatch study: model-specific FP8 recipes preserve OPD quality while providing ≥1.93× more KV-cache tokens than BF16. INT4 rollouts: possible, but model-dependent and needs the admission check.
- **Weight staleness.** If the rollout engine is $k$ steps behind the student, old-policy rollouts bias the loss. The fix is async with bounded staleness budget, not synchronous sync (too slow).

## Sources

- Paper: *DeepSeek-R1* — DeepSeek, 2025 — popularized R1-style distillation.
- Paper: *How Much Training-Inference Mismatch Can On-Policy Distillation Tolerate?* — Yu et al., 2026 — [arXiv:2611.11450](https://arxiv.org/abs/2611.11450)
