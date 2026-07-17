# Case Study: Ring-Zero (Ring-2.5-1T-Zero)

*The first public account of pure RLVR ("zero RL") training scaled to a **trillion-parameter** model — no long-CoT SFT cold start, no hand-crafted reasoning scaffolds, just outcome-verified RL on top of a base model. Ships with a training-stability recipe (clipped importance sampling, training/inference ratio correction, mixed-precision control) that makes zero-RL tractable at this scale, and documents a two-phase training dynamic — discovery, then sharpening — under which advanced reasoning behaviors emerge spontaneously.*

**Related concepts:** [zero-rl-scaling](../post-training/reasoning/zero-rl-scaling.md) · [clipped-importance-sampling](../post-training/clipped-importance-sampling.md) · [training-inference-ratio-correction](../post-training/training-inference-ratio-correction.md) · [rlvr](../post-training/rlvr.md) · [grpo](../post-training/grpo.md) · [long-cot-rl](../post-training/reasoning/long-cot-rl.md) · [partial-rollouts](../systems/partial-rollouts.md) · [fp8-training](../pre-training/fp8-training.md) · [deepseek-r1 case study](deepseek-r1.md) · [kimi-k1-5 case study](kimi-k1-5.md)

---

## What this is

**Ring-Zero (Ring-2.5-1T-Zero)**, from the Inclusion AI Ring team, is (per the authors) the first public run of the "zero-RL" recipe — reinforcement learning with verifiable rewards (RLVR) directly on a base model, with no CoT SFT priors — at **1-trillion parameters**. The paper (arXiv 2607.12395) is deliberately positioned as a *scaling experiment*: previous zero-RL efforts (R1-Zero at ~671B MoE / 37B active; academic replications at 3–70B) left open whether the recipe holds at frontier dense scale and what breaks along the way.

The answer: it holds, but you need specific training-stability fixes to get there, and the reasoning behaviors that emerge are *more* advanced than at smaller scale — not just more accurate. Ring-2.5-1T-Zero is competitive across seven challenging math benchmarks and produces cleaner, more structured reasoning traces than compressed baselines under a new CoT-quality rubric.

---

## The scaling context

Prior zero-RL evidence:
- **R1-Zero (DeepSeek, Jan 2025).** GRPO on DeepSeek-V3-Base (671B / 37B MoE), reaching 77.9% AIME24 pass@1. Established that reasoning behaviors emerge from pure RL.
- **Academic replications.** Small-scale (~3–70B dense) reproductions confirming the recipe but reporting fragility — training collapses, reward hacking via short outputs, non-monotonic curves.

Open questions before Ring-Zero:
1. Does zero-RL keep working at 1T+ dense parameters, or do off-policy variance and infrastructure friction become insurmountable?
2. If it works, does scaling improve *only* accuracy, or does the *character* of emergent reasoning change?
3. What algorithmic + systems changes are needed to keep the training loop stable?

Ring-Zero answers all three.

---

## The stability recipe

Three fixes that together make 1T zero-RL tractable:

### 1. Clipped importance sampling

Zero-RL runs off-policy — rollouts are generated with a snapshot of the policy, then used to update a policy that has since moved. At 1T, the drift between snapshot and current policy amplifies importance-sampling variance to the point where naive PPO/GRPO objectives blow up. Ring-Zero applies **clipped importance sampling** to bound the per-sample IS weight, keeping the gradient variance controlled without discarding the sample entirely.

Depth file: [clipped-importance-sampling](../post-training/clipped-importance-sampling.md).

### 2. Training-inference ratio correction

Rollouts are produced by an inference engine (vLLM-style, fused-kernel, FP16/FP8) whose numerics differ subtly from the training kernel (FP32 accumulate, different attention implementation). At small scale the divergence is noise; at 1T it compounds into a systematic bias in the rollout distribution. Ring-Zero introduces a **training-inference ratio correction** to reconcile the two, ensuring the RL loss is computed against the distribution the inference engine actually samples from.

Depth file: [training-inference-ratio-correction](../post-training/training-inference-ratio-correction.md).

### 3. Mixed-precision control

Ring-Zero specifies careful placement of FP8/BF16/FP32 across the RL loop — where to cast, where to accumulate, where to hold master weights. Without this, the loss surface becomes noisy enough that the model oscillates rather than progresses. Related: [fp8-training](../pre-training/fp8-training.md) for the general pretraining context; Ring-Zero extends the same hygiene into the RL loop.

Together, these three fixes replace the ad-hoc gradient clipping / learning-rate warmup that smaller runs get away with.

---

## Training dynamic — discovery, then sharpening

Ring-Zero documents a repeatable two-phase pattern:

**Phase 1 — Discovery.** Early training: the model explores a wide variety of CoT styles. Length increases, but so does variability — different rollouts on the same problem use different strategies (numeric grinding, symbolic manipulation, case analysis). Accuracy improves in fits and starts. Reward-signal noise is high.

**Phase 2 — Sharpening.** Mid-to-late training: the winning styles consolidate. Rollouts become more consistent in structure. Length stabilizes. Accuracy improves smoothly. The model has essentially converged on a reasoning *style*, then refines within it.

This dynamic — reminiscent of the "aha moment" reported in R1-Zero but framed as a global training-progress arc — is documented as reproducible and is one of the paper's core observations.

Depth file: [zero-rl-scaling](../post-training/reasoning/zero-rl-scaling.md).

---

## Spontaneously emergent behaviors

Ring-Zero reports that at 1T, the model develops without any hand-crafting:

- **Anthropomorphism** — internal monologue that reads like a person deliberating ("Let me think about this more carefully…").
- **Structured formatting** — headings, numbered steps, self-imposed lemmas.
- **Self-verification** — re-derivation of key intermediate results as a check.
- **Parallel reasoning** — pursuing multiple approaches simultaneously in the same CoT and comparing outcomes.
- **"Context anxiety"** — explicit acknowledgement of context-window pressure and self-management of trace length.

The authors' framing: at 1T, hand-crafted heuristics (prompt scaffolds, imposed reasoning templates) become *redundant*. The model invents better versions itself.

---

## Results and evaluation

**Headline results.** Ring-2.5-1T-Zero is competitive across seven challenging math benchmarks. The paper favors a demonstrated-parity framing over headline numbers vs. any single reference model.

**Structured CoT evaluation.** Beyond final-answer correctness, Ring-Zero proposes a three-dimension rubric for reasoning-trace quality:

- **Comprehensibility** — can a human follow the reasoning?
- **Reproducibility** — could someone re-derive the answer from the trace?
- **Efficiency** — how much token budget is spent per unit of useful reasoning?

Under this rubric, Ring-2.5-1T-Zero produces structured, concise traces — better than compressed baselines and better than smaller zero-RL runs.

---

## Why it matters

- **Frontier-scale confirmation of RLVR.** Pure RL on verifiable rewards continues to work — and gets *better* — at 1T. The recipe is not a small-model artifact.
- **Reasoning behaviors scale in kind, not just degree.** Self-verification and parallel reasoning at 1T were not seen at 70B. Suggests some capabilities are effectively locked behind scale + RL.
- **A shared training-stability playbook.** Clipped IS + training/inference ratio correction + mixed-precision control is now the de-facto template other trillion-parameter RL efforts will follow.
- **Argues against recipe complexity.** No long-CoT SFT cold start. No process reward models. No MCTS. The paper's implicit thesis: at 1T, the model outperforms your scaffolds — get out of its way.
- **A structured-CoT rubric.** Comprehensibility × reproducibility × efficiency is a portable evaluation frame others should adopt.

---

## Related concept files

- [rlvr](../post-training/rlvr.md) — the underlying paradigm.
- [grpo](../post-training/grpo.md) — the algorithm family that Ring-Zero's clipped-IS variant extends.
- [long-cot-rl](../post-training/reasoning/long-cot-rl.md) — the same regime as R1-Zero and Kimi k1.5.
- [deepseek-r1 case study](deepseek-r1.md) — the immediate predecessor at ~671B MoE.
- [kimi-k1-5 case study](kimi-k1-5.md) — a different long-CoT recipe (online policy mirror descent + length penalty).
- [fp8-training](../pre-training/fp8-training.md) — background for the mixed-precision hygiene Ring-Zero extends.
- [partial-rollouts](../systems/partial-rollouts.md) — the systems-level RL efficiency work Ring-Zero builds on.
