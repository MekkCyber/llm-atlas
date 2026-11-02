# Unified Training

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Collapse pretraining, mid-training, SFT, and RL into a *single* likelihood-based training algorithm. RL objectives are rewritten as maximum-likelihood fits to model-generated targets, so what distinguishes "stages" becomes just which distributions to mix at each step. One codepath, one objective, four knobs: static-vs-generated ratio, mixture weights, filtering, update order. Reported Pareto-wins over staged pipelines on accuracy-vs-compute, task coverage, and retention.

**Prereqs:** [mid-training](mid-training.md), [../post-training/_post-training](../post-training/_post-training.md), [../post-training/_rl](../post-training/_rl.md)
**Related:** [../post-training/rejection-sampling](../post-training/rejection-sampling.md), [../post-training/grpo](../post-training/grpo.md)

---

## What it is

A reframing of the staged "pretrain → mid-train → SFT → RL" pipeline that treats every training signal uniformly. The key equivalence: RL policy gradients can be rewritten as maximum-likelihood fits to a (reward-reweighted) model-generated target distribution. Under that equivalence:

- SFT = max-likelihood on a labeled dataset.
- RLHF/RLVR = max-likelihood on a model-generated-and-filtered dataset.
- Rejection-sampling fine-tuning = max-likelihood on a verifier-filtered sample.
- Pretraining = max-likelihood on raw web text.

They all have the same shape. The only differences are the data source and the filter.

## How it works

The unified trainer maintains a *mixture buffer* at each step:

```
mixture_t = { (D_static, w_1), (D_SFT, w_2),
              (model_generated_rollouts, w_3, filter), ... }
```

- Static datasets `D_static` are standard pretraining and mid-training corpora.
- Labeled SFT datasets are mixed in with their own weight.
- Model-generated rollouts are produced *during* training (not in a separate stage), optionally filtered by a verifier or reward model, and added to the mixture.

The optimizer sees a single batched max-likelihood loss on the mixture. Only one thing is being optimized at any step, but *what it's optimized on* evolves as the mixture weights do.

Scheduling the mixture weights `w_i` over time replaces the sharp "end pretraining / start SFT" transitions of the staged pipeline with smooth distributional shifts.

## Why it matters

- Staged pipelines leak capability between stages (SFT forgetting after RL, mid-training degrading SFT quality). A single-trainer formulation shares no clear mechanism for forgetting because there is no "transition" to a new objective.
- Operational: one codepath, one checkpoint cadence, one set of hyperparameters, one evaluation harness.
- The RL-as-MLE equivalence is a *simplification*, not an approximation — the paper proves the gradients match under reweighting. So the unified trainer gets no worse than its staged counterpart in theory, and strictly better in practice because mixture scheduling is a richer design space than sharp stage transitions.
- Shows a stronger starting point for *continual* learning: the final model is already a mixture-fitter, so adding a new target distribution is just another weight to raise.

## Gotchas & tricks

- Reward-reweighted MLE is numerically fragile when rewards vary over orders of magnitude; normalize per batch.
- Mixing on-policy rollouts with static text requires careful data loader design to avoid starving one stream.
- Verifier/reward availability matters — if you have no verifier, the "RL" slot of the mixture collapses to SFT on model-generated continuations, which is distillation on your own policy. Not a bad baseline, but not RL either.
- The paper reports Pareto wins at tested scales; frontier-scale generalization is still open.
- Compatible with existing techniques (LoRA, FP8, MoE) because the inner loop is still just MLE.

## Sources

- Paper: *Training Foundation Models with a Single Algorithm* — Song, Zhang, Tajwar, Arora, Schneider, Khashabi, Salakhutdinov, Bagnell, Singh, Zanette, CMU / JHU, 2026 — [arXiv:2610.40564](https://arxiv.org/abs/2610.40564).
- Related: *Rejection Sampling Fine-Tuning* (Llama 2 Chat) — one special case of the mixture.
- Related: *GRPO* — DeepSeek, 2024 — another special case under the reweighting view.
