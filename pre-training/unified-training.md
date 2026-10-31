# Unified Training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reframing of foundation-model training that puts pretraining, mid-training, SFT, and RL under a *single* likelihood-based algorithm. The key step is deriving the RL objectives as maximum-likelihood fitting to model-generated target distributions. Under that frame, every "stage" is "train on a data mixture," and the open question becomes **which data, in what proportion, when** — not which algorithm. Staged training (pretrain → SFT → RLHF → RLVR) is a special case; a trainer that mixes static corpora with on-policy RL rollouts in one loop is another.

**Prereqs:** [_post-training](../post-training/_post-training.md), [_rl](../post-training/_rl.md)
**Related:** [mid-training](mid-training.md), [rlvr](../post-training/rlvr.md), [grpo](../post-training/grpo.md)

---

## What it is

Modern LLM training is a chain of nominally distinct stages:

1. **Pretraining** — likelihood on static corpora.
2. **Mid-training** — likelihood on curated / rebalanced corpora.
3. **SFT** — likelihood on human/model-written demonstrations.
4. **RL** — policy gradient on generations against a reward.

The unified-training view: for a class of RL objectives (verifiable rewards, preference-derived rewards expressible as a KL-regularized MLE), the RL gradient *is* a maximum-likelihood gradient — just against a model-generated target distribution rather than a static one. Every stage becomes "fit likelihood on data drawn from distribution $\mathcal{D}_i$ at mixing weights $\alpha_i(t)$." Staged training is the special case where only one $\alpha_i$ is non-zero at a time.

## How it works

- **One trainer, one loss.** A single likelihood-based optimizer runs the whole pipeline. The "RL" sources are just streams of (prompt, response, scalar) tuples that feed the same loss.
- **Data mix is scheduled.** Instead of discrete stages with discrete transitions, the design problem is $\alpha_i(t)$: how much static data, how much mid-training-style curated data, how much on-policy RL at each step.
- **Co-training static + on-policy.** In the paper's instantiation, the trainer mixes a static corpus with RL generations. The RL side contributes the capability push; the static side preserves broader capabilities and prevents the staged-training regression where SFT/RL erodes pretrained knowledge.
- **Continual-learning starting point.** A unified-trained checkpoint has broader basin of capabilities than a staged-trained one at matched compute, making continued learning cheaper.

## Why it matters

The paper's claim is reframing more than a specific new recipe. If the "stages are not fundamental" view holds up:

- The design question moves from "when do we switch stages" to "what is the data-mixture schedule."
- Infrastructure (data loaders, RL rollout workers, SFT batches) can share one trainer loop.
- Capability-preservation issues at stage transitions (SFT regressing pretraining knowledge, RL regressing SFT behavior) are addressed by continuous mixing rather than per-stage mitigations.
- Pretraining and post-training stop being separate codepaths and become schedule hyperparameters.

Empirically, unified training is reported to improve the accuracy–compute tradeoff, improve task coverage, preserve broader capabilities, and give a stronger starting point for continual learning.

## Gotchas & tricks

- **Scope of the "unification"**: holds for RL objectives expressible as MLE against a model-generated target (verifiable-reward RL, KL-regularized preference optimization). More exotic reward signals may require extension.
- **Data-mix scheduling is now the whole job.** You trade one hard problem (stage transitions) for a different hard problem (continuous mixing weights). Not clearly easier a priori.
- **On-policy RL inside the main loop needs rollout infrastructure during pretraining-scale runs.** Non-trivial — fits better in labs that already serve the model from the trainer (vLLM-style co-location).
- **Baseline comparisons matter.** The paper reports gains over staged training; apples-to-apples comparisons depend on matching the data mixtures seen by each run.

## Sources

- Paper: *Training Foundation Models with a Single Algorithm* — Y. Song, A. Zhang, F. Tajwar et al., CMU · JHU · Aurora Innovation, 2026 — arXiv:2610.40564.
