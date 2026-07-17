# Zero-RL Scaling Dynamics
*Depth — what happens to zero-RL training as parameter count scales into the trillions: two-phase dynamics, spontaneous reasoning behaviors, and what stops working from the smaller-scale playbook.*

**TL;DR:** Zero-RL — RLVR directly on a base model without any long-CoT SFT — was popularized by R1-Zero at ~671B MoE and reproduced at smaller scales with varying stability. Ring-Zero (Inclusion AI, 2026) reports the first 1T-parameter zero-RL run and documents a repeatable dynamic: a **discovery phase** where the model explores diverse CoT styles, followed by a **sharpening phase** where the winning style consolidates and accuracy improves smoothly. At 1T, advanced reasoning behaviors emerge spontaneously — self-verification, parallel reasoning, structured formatting, "context anxiety" — rendering hand-crafted scaffolds redundant. Naive scaling suffers from poor readability, token redundancy, and lack of adaptive reasoning depth; the fix is a stability recipe (clipped IS + training-inference ratio correction + mixed-precision control), not more prompt engineering.

**Prereqs:** [rlvr](../rlvr.md), [long-cot-rl](long-cot-rl.md)
**Related:** [../clipped-importance-sampling](../clipped-importance-sampling.md), [../training-inference-ratio-correction](../training-inference-ratio-correction.md), [../../case-studies/ring-zero.md](../../case-studies/ring-zero.md), [../../case-studies/deepseek-r1.md](../../case-studies/deepseek-r1.md)

---

## What it is

"Zero RL" refers to the recipe of running reinforcement learning with verifiable rewards **directly on a base model** — no SFT cold start, no CoT demonstrations, no hand-crafted reasoning templates. R1-Zero (Jan 2025) demonstrated it worked at ~671B MoE / 37B active with GRPO + rule-based rewards, producing autonomously emerging reasoning behaviors ("aha moments," length growth, self-reflection).

**Zero-RL scaling** is the study of what changes as parameter count grows into the trillions. The interesting questions:

1. Does the recipe keep working, or does off-policy variance / infrastructure friction break it?
2. Does scaling just improve accuracy, or does the *character* of emergent reasoning change qualitatively?
3. What additional engineering (algorithmic, systems, numerical) is required at scale that smaller runs got away without?

## How it works

### Two-phase training dynamic

Ring-Zero documents a repeatable pattern at 1T:

**Phase 1 — Discovery.** Early training. The model explores a variety of CoT styles: symbolic manipulation, numeric grinding, case analysis, various formatting choices. Rollouts on the same prompt use different strategies. Length grows autonomously but variably. Accuracy improves in bursts. Reward signal is high-variance.

**Phase 2 — Sharpening.** Mid-to-late training. Successful styles consolidate. Rollouts become more consistent in structure across the same prompt. Length stabilizes to a task-appropriate range. Accuracy improves smoothly. The model has effectively "picked" a reasoning style and now refines within it.

This is a global training-progress arc, distinct from the single-run "aha moment" phenomenology in R1-Zero. It reproduces across training runs.

### Spontaneous behaviors that emerge at 1T

Without any prompt scaffolding or reward shaping:

- **Self-verification** — re-derivation of intermediate results as a check ("Let me verify this claim by …").
- **Parallel reasoning** — the model pursues multiple approaches simultaneously in one CoT and compares outcomes.
- **Structured formatting** — headings, numbered steps, self-imposed lemmas.
- **Anthropomorphism** — internal deliberation phrased as if to another person.
- **"Context anxiety"** — the model explicitly acknowledges context-window pressure and manages trace length.

Several of these were absent or weak at 70B–200B zero-RL runs. This is one of Ring-Zero's core empirical claims: the qualitative reasoning repertoire *itself* scales, not just the accuracy.

### What breaks under naive scaling

Ring-Zero identifies three failure modes when scaling zero-RL without the stability recipe:

1. **Poor readability.** Long traces devolve into stream-of-consciousness with no structure.
2. **Token redundancy.** Reasoning steps repeat, restate, and pad without new content.
3. **Lack of adaptive reasoning depth.** The model uses similar length for easy and hard problems — no allocation.

These are addressed not by prompt engineering but by three training-loop fixes:

- **[Clipped importance sampling](../clipped-importance-sampling.md)** — bounds per-sample IS variance at large off-policy drift.
- **[Training-inference ratio correction](../training-inference-ratio-correction.md)** — reconciles inference-engine numerics with training-kernel numerics.
- **Mixed-precision control** — careful FP8/BF16/FP32 placement across the RL loop (see [fp8-training](../../pre-training/fp8-training.md) for the general context).

## Why it matters

- **Confirms zero-RL at frontier scale.** The recipe is not a small-model artifact and gets *better* with scale.
- **Reasoning behaviors scale in kind, not just degree.** Some capabilities appear only past a threshold — a strong argument against "just distill from a larger reasoner" strategies for those capabilities specifically.
- **Argues against recipe complexity at scale.** Hand-crafted scaffolds become redundant; the model out-invents them. This is a real signal to reduce, not add, complexity as models grow.
- **Establishes a portable stability playbook.** Clipped IS + training-inference ratio correction + mixed-precision control is now the reference template for other 1T+ RL efforts.

## Gotchas & tricks

- **Discovery phase looks broken.** Loss is noisy, accuracy oscillates, style varies wildly — this is the *point* of discovery, not a failure. Don't kill runs prematurely; the sharpening phase is the payoff.
- **Length growth is not the goal.** At smaller scale, unbounded length growth is a symptom of missing length regularization. At 1T with the stability recipe, length growth *followed by stabilization* is the healthy pattern.
- **Reward variance is diagnostic.** Reward variance should be high in discovery and lower in sharpening. Persistent high variance late in training usually indicates IS clipping issues or numerical mismatch.
- **Behavior emergence is empirical, not guaranteed.** Ring-Zero saw these specific behaviors at 1T with their recipe; other recipes may see different behaviors. Log-and-observe, not assume.
- **The recipe isn't optional at scale.** Small-scale zero-RL runs succeed without the stability recipe; the belief that "1T just needs a bigger machine" is wrong. Skipping the fixes produces the three failure modes.
- **CoT-quality evaluation matters.** Final-answer accuracy hides the readability / redundancy failures. Adopt structured-CoT metrics (comprehensibility, reproducibility, efficiency) alongside benchmarks.

## Sources

- Paper: *Ring-Zero: Scaling Zero RL to a Trillion Parameters for Emergent Reasoning* — Cao, Liu, Zhan, Lan, Li, Yan, Peng, Dong, Zhang, Wang, Kong, Wen, Zhao, Zhang, Zhou, 2026 — [arXiv 2607.12395](https://arxiv.org/abs/2607.12395). Introduces the discovery/sharpening framing and the 1T stability recipe.
- Paper: *DeepSeek-R1* — DeepSeek-AI, 2025 — arXiv 2501.12948. The prior best-known zero-RL run at ~671B MoE / 37B active.
