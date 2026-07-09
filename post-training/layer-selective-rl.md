# Layer-selective RL post-training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RL post-training gains are highly **concentrated in a small subset of transformer layers** — in many cases a single middle-stack layer recovers most of the gains, sometimes surpassing full-parameter RL. Layer-aware training (freeze all but the top-$k$ layers by contribution) and ensembles of layer-specialized models both beat standard full-parameter RL across Qwen2.5 / Qwen3 with GRPO / GiGPO / Dr.GRPO on math, code, and agentic tasks.

**Prereqs:** [grpo.md](./grpo.md), [_rl.md](./_rl.md)
**Related:** [fine-tuning/README.md](./fine-tuning/README.md), [../interpretability/README.md](../interpretability/README.md)

---

## What it is

Full-parameter RL updates every layer of the transformer, implicitly assuming layers contribute similarly to gains. This paper's systematic ablation shows they don't: define **layer contribution** = fraction of full-RL improvement recovered when only that one layer is trained. Rank layers by contribution across 7 models, 3 algorithms, and multiple tasks. Pattern: middle layers dominate, edge layers barely matter, and rankings *transfer* across datasets, tasks, model families, and RL algorithms.

## How it works

**Diagnosis step.** For a target model + algorithm + task:

1. Run a "single-layer probe" — train only layer $\ell$ (freeze all others), run the same RL loop for the same total tokens/rollouts.
2. Measure the improvement over the SFT baseline.
3. Divide by the improvement full-parameter RL achieves. That ratio is layer $\ell$'s contribution.

Sweep over all $\ell$ → a contribution ranking.

**Training strategies** derived from the ranking:

- **Top-$k$ layer-selective RL.** Train only the top-$k$ layers by contribution; freeze everything else. Cheaper compute (fewer parameters have grads / optimizer state) and often *higher* final quality.
- **Layer-specialized ensembles.** Train separate models each with a different specialized-layer configuration, ensemble at inference. Complementary behaviors yield additional gains.

The ranking is stable enough that you can (a) compute it once on a small proxy setup, (b) apply the resulting layer selection to your real RL run.

## Why it matters

- **Practical: dramatically cheaper RL post-training.** Freezing all-but-one-layer means most parameters carry no optimizer state, no gradient computation — RL step cost drops accordingly. GRPO's already cheap-relative-to-PPO advantage compounds.
- **Mechanistic: RL modifies a small subset of layers.** This is direct evidence for the "RL doesn't add capabilities, it elicits them" hypothesis — if the gains were about instilling new knowledge, you'd expect a distributed update. Middle-layer concentration hints at where reasoning / decision-making behaviors are localized.
- **Transfer implication.** Rankings correlate across algorithms and tasks → layer contribution is a property of the *pretrained model*, not the RL specifics. Interp-adjacent finding.
- **Ensemble opportunity.** Layer-specialized ensembles give complementary gains → different layers, when trained in isolation, learn different aspects.

## Gotchas & tricks

- **Ranking depends on the SFT initial checkpoint.** A different SFT starting point can reorder layers. The middle-layer bias is robust; the exact rank is not.
- **"Single layer suffices" doesn't mean "one layer is enough forever."** For maximal quality use the top-$k$ layers or the ensemble. Single-layer training is the striking baseline, not the recommended default.
- **KL to reference model still matters.** Freezing most layers doesn't remove the need for KL regularization; the trained layer can still overfit to the reward signal.
- **Not a substitute for good SFT.** The paper's setup starts from strong SFT checkpoints; the RL update is a small polish. If your SFT is weak, layer-selective RL will inherit its weakness.
- **Composes with LoRA/QLoRA.** Layer selection and parameter-efficient adapters are orthogonal — layer-selective LoRA-RL is a natural next step.

## Sources

- Paper: *Is One Layer Enough? Training a Single Transformer Layer Can Match Full-Parameter RL Training* — Zhang, Hu, Glentis, et al., University of Minnesota / Peking University / Amazon, 2026 — [arXiv:2607.01232](https://arxiv.org/abs/2607.01232).
