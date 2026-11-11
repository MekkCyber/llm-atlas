# Training-Inference Mismatch Budget
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Treat quantization drift, weight staleness, and injected noise in the rollout engine as additive on a single distribution-shift scale, calibrate how much the current model can tolerate, and screen rollout configs with a lightweight **admission check** before paying for a full training run. For on-policy distillation specifically, model-specific FP8 recipes match BF16 quality while unlocking ≥1.93× more KV-cache tokens; FP8 + async reaches target quality 1.62× sooner than synchronous BF16.

**Prereqs:** [./on-policy-distillation.md](./on-policy-distillation.md), [../quantization/fp8.md](../quantization/fp8.md)
**Related:** [../systems/partial-rollouts.md](../systems/partial-rollouts.md), [../pre-training/fp8-training.md](../pre-training/fp8-training.md), [../quantization/_number-formats.md](../quantization/_number-formats.md)

---

## What it is

On-policy distillation (OPD) and online RL both sample rollouts from a rollout engine that is *not bit-identical* to the model being trained. The engine may be quantized (FP8, INT4), may be a few steps behind (async), and inevitably injects some numerical noise (reduction order, kernel choice). Each of these shifts the rollout distribution away from the student's "true" sampling distribution.

Prior work handled each axis separately (FP8 ablations, staleness bounds, noise injections). The mismatch-budget framing is: **put all three on the same scale**, call the aggregate "mismatch", and empirically measure how much a given model can absorb before OPD quality degrades.

## How it works

1. **Measure the mismatch scale** between a candidate rollout config and the reference (full-precision, synchronous). The paper uses a KL-based divergence between rollout-engine samples and reference samples on a calibration prompt set; other reasonable metrics include forward-KL of per-token distributions.
2. **Fit a model-specific budget curve** — how OPD final quality (eval loss or downstream metric) degrades as mismatch grows. The curve has a plateau region (OPD indifferent to mismatch) and a cliff.
3. **Admission check.** For any new rollout-engine config (e.g. "FP8 weights + INT8 activations + async k=4"), measure its mismatch scale against the reference *once* and compare to the fitted budget. If under budget, admit; otherwise reject without running a full training job.

**Finding:** OPD's budget is substantially larger than online RL's; some layers dominate the budget consumption (and should stay in higher precision), others are indifferent. This suggests *per-layer* rollout quantization recipes rather than uniform ones.

## Why it matters

- Converts rollout-engine configuration from guesswork into a resource-allocation problem.
- Empirically justifies the "FP8 rollout + BF16 train" split many modern distillation pipelines already use informally — and quantifies when INT4 crosses the cliff.
- Model-specific FP8 recipes **≥1.93× KV-cache tokens** vs BF16 at matched quality — a direct throughput win for long-context post-training.

## Gotchas & tricks

- **Budget is model-specific, not architecture-specific.** The same architecture at different training stages can have very different cliffs. Re-calibrate per checkpoint.
- **Calibration set choice matters.** A narrow calibration set hides mismatch that would show up on real training prompts. Use a diverse subset.
- **Weight staleness and quantization combine.** Each is under budget alone; stacked, they can exceed it. The admission check should score the full stack together.
- **Online RL budget ≪ OPD budget.** Don't port the OPD admission thresholds over to PPO/GRPO — the RL loss is much more sensitive to the sampling distribution.
- **Sensitive layers are often the output layer.** Keep the final logit projection in higher precision.

## Sources

- Paper: *How Much Training-Inference Mismatch Can On-Policy Distillation Tolerate?* — Yu, Wang, Jiang, Zhang, Chai, Jiang, Yoneki, Shi, 2026 — [arXiv:2611.11450](https://arxiv.org/abs/2611.11450) — code at <https://github.com/benyucong/train-inference-mismatch-opd/>.
