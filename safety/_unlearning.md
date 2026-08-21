# Unlearning

*Taxonomy — post-hoc removal of specific knowledge or behaviours from a trained LLM without full retraining.*

**TL;DR:** LLM unlearning objectives remove targeted facts, capabilities, or behaviours from a model already trained on them. Every method fights the same two forces — **residual memorisation** (facts leak back under paraphrase, translation, or membership queries) and **catastrophic forgetting** (loss of unrelated capabilities). Modern approaches sit on a spectrum from **gradient ascent on the forget set** (fast, unstable) through **negative-preference optimization** (DPO-flavoured, more stable) to **re-alignment / retention regularisation** (paired forget+retain losses). Popularity- or frequency-weighted variants like AdaPop (2026) recognise that "forget one fact" isn't a single objective; popular facts resist removal exponentially harder than rare ones.

**Related taxonomies:** [_attacks.md](./_attacks.md) (attacks that recover unlearned content), [_jailbreaks.md](./_jailbreaks.md) (attacks that bypass refusal generally).
**Depth files covered here:** [adapop-unlearning](adapop-unlearning.md)

---

## The problem

A production LLM has learned things you now need it not to know or not to produce: copyrighted passages, private personal data, an unsafe capability, an obsolete API. Full retraining is prohibitive; deleting weights is untargeted; refusing at inference time (a filter) is not the same as forgetting. Unlearning aims to change the *weights themselves* so that:

- The **forget set** $D_f$ can no longer be reproduced or completed.
- The **retain set** $D_r$ (everything else the model should still do) is preserved.
- The output distribution on unlearned queries approaches that of a model that never saw $D_f$.

Every practical method compromises on at least one of these axes.

## The shared pattern

All variants minimise a two-part objective of the form

$$\mathcal{L}_{\text{unlearn}} = \mathcal{L}_{\text{forget}}(D_f) + \lambda\,\mathcal{L}_{\text{retain}}(D_r)$$

They differ in **what** $\mathcal{L}_{\text{forget}}$ is (negative log-likelihood, DPO-style preference, gradient ascent, task-vector subtraction, popularity-weighted variants) and **how** $\mathcal{L}_{\text{retain}}$ is expressed (KL to a reference model, replay on a retain corpus, low-rank constraint).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Gradient ascent on $D_f$ | Flip the sign of NLL on the forget set | Extremely unstable — model diverges | Rarely used alone; a building block |
| Negative Preference Optimization (NPO) | DPO with the forget completions as the "rejected" side | Stable; slower forgetting | Default modern baseline for factual forget |
| Task-vector subtraction | Subtract the delta of "trained on $D_f$" from current weights | Requires a paired checkpoint | Model-editing settings |
| Retention-regularised NPO | NPO + explicit KL / replay on $D_r$ | Extra compute per step | When retain quality matters most |
| [adapop-unlearning](adapop-unlearning.md) (AdaPop) | Popularity-weighted forget loss with token-confidence gating | Needs a popularity estimate | Heterogeneous forget sets where a few facts dominate |

## How to choose

- **Default** in 2026: NPO with a retain-set KL regulariser. Well-understood, moderately fast, doesn't blow up.
- If your forget set has **very unequal popularity** (some facts are ubiquitous, others rare): AdaPop-style popularity weighting equalises effective removal without hand-tuning per fact.
- If you can afford to **retrain from a checkpoint** and only need to remove a scoped delta: task-vector subtraction is exact when the paired checkpoint exists.
- Always evaluate against **paraphrase, translation, and membership-inference attacks** — surface-level forgetting is trivially defeated by rephrasing.

## Adjacent but distinct

- **Refusal training / safety RLHF** — makes the model *decline* the query; doesn't remove the knowledge. Complementary, not substitute.
- **Alignment via constitutions** — changes preferred outputs without touching underlying knowledge.
- **Differential privacy at training time** — a *prevention* strategy for pretraining; unlearning is a *remediation* strategy afterwards.
- **Model editing** (ROME, MEMIT) — targeted *replacement* of a fact rather than removal; the eval and objective are different.

## Sources

- Paper: *The More Popular, The Harder to Forget: Adaptive Popularity for LLM Unlearning* — Borisiuk et al., AIRI / Skoltech, 2026 — [arXiv 2608.14229](https://arxiv.org/abs/2608.14229) — the popularity-weighted variant and the "popularity-of-fact vs unlearning difficulty" observation this taxonomy anchors on.
- Paper: *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning* — Zhang et al., 2024 — the NPO baseline.
- Benchmark: *TOFU: A Task of Fictitious Unlearning for LLMs* — Maini et al., 2024 — the standard unlearning eval.
