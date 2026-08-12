# u-OPSD — Unsupervised On-Policy Self-Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy self-distillation with **no external supervision at all** — no ground truth, no verifier, no larger teacher. Sample `n` rollouts, build a pseudo-solution by majority vote under a self-consistency threshold, then distill the model onto the *disagreeing* completions conditioned on that pseudo-solution. Matches or beats GRPO on math (AIME24/25, HMMT25, MATH500, AMC23) while needing zero labels or verifiers.

**Prereqs:** [on-policy-distillation.md](on-policy-distillation.md), [grpo.md](grpo.md), [rejection-sampling.md](rejection-sampling.md)
**Related:** [rlvr.md](rlvr.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [spot-distillation.md](spot-distillation.md)

---

## What it is

Standard post-training assumes access to *some* supervision: ground-truth labels (SFT), verifiable rewards (RLVR / GRPO), a teacher distribution (OPD), or human preferences (RLHF / DPO). u-OPSD asks: can a model improve at reasoning using only its own outputs, no external signal?

Yes — under one assumption: for many reasoning tasks, the model's *majority-vote* answer across `n` rollouts is close enough to correct that it's a useful proxy target. Self-consistency at high enough `n` reveals the model's own most-confident answer, which is often right even when individual rollouts are wrong.

## How it works

1. **Rollout.** Sample `n` completions per prompt: `{o_1, ..., o_n} ∼ π_θ(· | q)`. Typical `n = 8–16`.
2. **Pseudo-solution construction.** Extract the final answer from each `o_i`, tally votes. If the majority answer's vote share exceeds a **self-consistency threshold** `τ` (e.g. τ = 0.5), take that as the pseudo-solution `y*`. Prompts where consensus is below `τ` are skipped for this step — the model is too uncertain to teach itself.
3. **Disagreement split.** Partition the rollouts into agreeing (`o_i` ends with `y*`) and disagreeing (`o_i` ends with something else).
4. **Conditioned distillation.** Compute the teacher distribution as the model's own distribution *conditioned on `y*`* — i.e. `π_θ(· | q, [y*-hint])` for some form of hinting or answer-anchoring. Distill on the **disagreeing completions** by minimizing per-token KL to this conditioned distribution.
5. **Iterate.** Repeat with fresh rollouts each step.

The intuition: on disagreeing rollouts, the model was confidently wrong. Anchoring the target on the majority-vote pseudo-solution provides the corrective signal — no external verifier needed to say which completion was right.

## Why it matters

- **Removes the last supervision requirement.** RLVR needs verifiable answers; SFT needs labels; OPD needs a teacher. u-OPSD needs only the model. Opens continued improvement in domains where verifiers don't exist and labels are prohibitive.
- **Beats or matches supervised baselines.** On Qwen3 non-thinking mode: +8.5% (4B) / +10.7% (8B) over base across five math benchmarks. Beats supervised OPSD by +3.2% / +2.3%. Beats GRPO by +0.7% / +1.1%.
- **Reveals a "self-teachable" regime.** The technique only works when the model's majority-vote answer is meaningfully more accurate than a random sample — a nontrivial condition on the base model's capability. Below that threshold, u-OPSD collapses (the pseudo-solution is noise).
- **Composable.** u-OPSD is a *supervision constructor*, not a full training algorithm. It can substitute for the label / reward in any downstream loss (SFT, OPD, GRPO).

## Gotchas & tricks

- **Threshold `τ` sets a capability filter.** Too low → noisy pseudo-solutions poison training; too high → most prompts get dropped and the training set shrinks. `τ = 0.5` is the paper's default.
- **Fails on tasks the base model is uniformly wrong on.** If majority vote is worse than random, u-OPSD amplifies the wrong answer. Sanity check with a small verified set periodically.
- **Requires diverse rollouts.** Deterministic-ish decoding (very low temperature) gives near-identical completions — no disagreement, no signal. Sample at nonzero temperature (e.g. 0.7–1.0).
- **Answer extraction is a hidden dependency.** Majority vote needs a canonical answer form (numeric, multiple-choice tag, code block). Free-form outputs need a separate extraction step; noise there caps the whole loop.
- **Combines cleanly with GRPO.** u-OPSD can seed GRPO's rollouts, or u-OPSD's pseudo-solutions can act as the "correct answer" in RLVR when no verifier exists.

## Sources

- Paper: *On-Policy Self-Distillation without Any Supervision* — Li, Wang, Liang, Tian, Fu, Vasconcelos, 2026 — the source and empirical results.
- Related: [on-policy-distillation.md](on-policy-distillation.md) for the OPD/OPSD family this extends.
- Related: [rejection-sampling.md](rejection-sampling.md) — majority-vote is a softer variant of the reject-unless-verified idea.
- Code: https://github.com/williamium3000/u-opsd
