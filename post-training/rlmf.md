# RLMF — Reinforcement Learning with Metacognitive Feedback
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RLMF uses the model's **self-judgment about the quality of its own completions** as the reward signal for preference optimization. No separate reward model, no external judge; the LLM is trained to grade its own outputs, and those grades are used both to rank preference pairs and to select high-value training examples. Applied to faithful calibration (aligning expressed with intrinsic uncertainty), RLMF beats standard RL by up to 63% and reaches SOTA on faithful-calibration benchmarks while preserving accuracy.

**Prereqs:** [dpo](dpo.md), [_rewards](_rewards.md)
**Related:** [rlvr](rlvr.md) · [ppo](ppo.md) · [../safety/cot-monitoring](../safety/cot-monitoring.md)

---

## What it is

RLHF and RLVR both extract training signal from *outside* the model: a human preference, a rule-based verifier, a learned reward model. RLMF (Liu et al., 2026) extracts it from *inside*: the same LLM being trained provides metacognitive self-judgments about the quality of its own completions, and those judgments drive preference optimization.

Two mechanisms:

1. **RLMF proper.** Refine completion rankings during preference optimization using the model's self-judgment of completion quality. Preference pairs are (re-)labeled by the model itself.
2. **Metacognitive data selection.** Use those same self-judgments to identify high-value training examples — a form of active learning that beats naive uncertainty-based selection.

## How it works

### Self-judgment as a reward source

For a prompt $q$ and two candidate completions $o^+, o^-$, query the model with a metacognitive prompt: "How well does this response solve the task?" The model returns a scalar (log-prob of a rating token, or a numeric score). Use these self-scores to construct the preference pair used by DPO/PPO. The self-judgment prompt and the generation prompt use the same weights; there is no separate reward model.

### Two-stage decoupled pipeline for faithful calibration

Faithful calibration (FC) — the property that a model's *expressed* confidence matches its *intrinsic* uncertainty — is itself a metacognitive task. RLMF splits it:

- **Stage A — calibrate internal scores.** Use RLMF to align the model's self-reported confidence with correctness.
- **Stage B — map to language.** Use targeted output editing to convert calibrated numeric confidence into natural, context-adaptable linguistic uncertainty ("I'm not sure but ...").

### Metacognitive data selection

Score every candidate training example by the *variance* or *entropy* of the model's own self-judgments over multiple samples. High disagreement → high-value example. Beats naive active-learning heuristics because it selects examples where the model is genuinely uncertain about its own performance.

## Why it matters

- **Bypasses the reward-model bottleneck.** No separate RM to train or maintain; no RM drift; no RM overfitting. The model *is* the RM.
- **Calibration is a first-class training target.** Historically calibration was measured post-hoc and patched with temperature scaling. RLMF makes it a training objective.
- **Compatible with any base LLM.** No architecture change; no new pretraining. Any model that can be prompted to self-judge can be RLMF-tuned.
- **Bridges to safety.** Faithful uncertainty expression is a foundational safety property: it lets deployers trust the model's "I don't know". RLMF is a mechanism for producing it at scale.

## Gotchas & tricks

- **Self-judgment reliability floor.** If the model's self-judgment is worse than random, RLMF injects noise. Empirical rule of thumb: self-judgment AUC ≥ 0.7 on a held-out task before RLMF is worth running on that task.
- **Judge-bias drift.** The model may develop idiosyncratic self-scoring preferences and reward-hack them. Mix in an external verifier on verifiable subtasks (math answers, unit tests) as a periodic sanity check.
- **Decouple confidence from language.** Optimizing "sound uncertain" text directly leads to verbal hedging without calibration. The two-stage decoupled pipeline is important — calibrate internally first, then map to language.
- **Composes with RLVR.** For verifiable-reward tasks, mix the RLMF self-judgment reward with a rule-based verifier reward. The verifier grounds the model against reward hacking; the self-judgment gives dense signal on non-verifiable subtasks.

## Sources

- Paper: *Reinforcement Learning with Metacognitive Feedback Elicits Faithful Uncertainty Expression in LLMs* — Liu, Caciularu, Yona, Szpektor, Cohan, 2026 — Yale / Google Research; the RLMF paper.
