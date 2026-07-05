# TAP — Task-Agnostic Pretraining for Vision-Language-Action

*Depth — decouple motor-skill pretraining (self-supervised inverse dynamics on unlabeled interaction data) from language-conditioned task grounding for VLA models.*

**TL;DR:** VLAs plateau because labeled expert demonstrations are scarce and expensive. **TAP** splits the pretrain into two stages: (1) **learn to move** — self-supervised inverse dynamics on *unlabeled* interaction data, no language, no task labels; (2) **learn to do** — ground those motor skills in language with a small labeled set. On SIMPLER, TAP matches models trained on **>1M expert trajectories** with far fewer labels; **+10% absolute** over standard behavior cloning at matched labels. Real-world WidowX experiments hold **25% success** under camera perturbations where internet-scale baselines drop to zero.

**Prereqs:** [README.md](README.md)
**Related:** [../pre-training/mid-training.md](../pre-training/mid-training.md)

---

## What it is

A two-stage pretraining recipe for Vision-Language-Action (VLA) foundation models. The first stage develops **generic motor priors** without needing any language or task labels; the second stage grounds those priors in natural-language task descriptions with modest label budget. The decoupling exploits the fact that unlabeled interaction data is far cheaper than labeled expert demonstrations.

## How it works

### Stage 1 — self-supervised inverse dynamics

Given consecutive observations $(o_t, o_{t+1})$, train the model to predict the action $a_t$ that connects them: $\hat{a}_t = f_\theta(o_t, o_{t+1})$. Data comes from any interaction stream — teleoperation, prior policy rollouts, human demos ignoring the language labels, sim data. Language and task descriptions are not used. The model learns the **motor manifold** of the embodiment: what actions cause what visual transitions.

### Stage 2 — language-conditioned task grounding

With the motor prior in place, add a small labeled dataset (task descriptions + demonstrations). Train the model to condition its action prediction on the language. Because the motor prior already encodes plausible action space, the language grounding only has to *pick* actions, not learn them from scratch — hence the small label budget.

### Why this decoupling works

Standard VLA pretraining conflates "what does this scene afford as actions?" (motor) with "which of those actions matches this instruction?" (grounding). If both must be learned from the same labeled dataset, label scarcity caps both. Stage 1 answers the first question from cheap unlabeled data; stage 2 focuses labels on the second.

## Why it matters

- **VLA label scarcity is *the* bottleneck.** The field has been throwing internet-scale data at VLAs to compensate; TAP shows a much cheaper path — treat motor pretraining as its own stage with its own (unlabeled) data source.
- **Robustness under perturbation.** The 25% success under camera-perturbation on WidowX (vs 0% for internet-scale baselines) is the kind of transfer signal deployed robots actually need. Internet-scale baselines overfit to canonical camera geometries; motor pretraining doesn't.
- **Modular VLA training.** Stage 1 can be shared across teams and embodiments; stage 2 is task/language-specific and cheap. This is closer to how NLP pretraining decoupled from task-specific fine-tuning.

## Gotchas & tricks

- **Inverse dynamics is embodiment-specific.** The motor prior learned on WidowX doesn't transfer to a Franka Emika arm without adaptation. Plan on a per-embodiment stage 1.
- **Unlabeled data quality still matters.** Random-policy exploration data gives a weaker motor prior than teleoperation rollouts. Whatever unlabeled data source you use, it should span the action space you care about.
- **Language-grounding data mix.** Too little stage-2 data and the model produces plausible motor primitives that ignore the instruction. Too much and you're back to conventional VLA training. The paper reports label-budget curves.
- **Camera perturbation robustness comes from stage 1.** Ablating stage 1 while keeping stage 2 collapses the perturbation robustness. The result is not a stage-2 phenomenon.

## Sources

- Paper: *Learning to Move Before Learning to Do: Task-Agnostic Pretraining for VLAs* — Shi, Wang, Yu, Ji, Gong, Qiu (Fudan University / Shanghai Innovation Institute), 2026 — [arXiv:2607.02466](https://arxiv.org/abs/2607.02466).
- Benchmark: SIMPLER (simulated manipulation policy learning benchmark), WidowX (real-world manipulation).
