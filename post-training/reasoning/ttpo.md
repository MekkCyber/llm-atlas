# Test-Time Policy Optimization (TTPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A labelless test-time training objective for LLM reasoning. On unlabeled prompts, sample a group of rollouts, take the majority-vote answer as a pseudo-label, then apply an **asymmetric** two-branch update: on-policy self-distillation for rollouts that *agree* with the vote, grouped-RL penalties on rollouts that *disagree*. Token-level gating protects both branches from noisy pseudo-labels: distillation ignores already-converged tokens, RL penalizes only *confidently wrong* ones. Matches label-supervised OPSD on five competition math benchmarks with no labels at all.

**Prereqs:** [../grpo.md](../grpo.md), [../rlvr.md](../rlvr.md), [../_rl.md](../_rl.md)
**Related:** [long-cot-rl.md](long-cot-rl.md), [../rejection-sampling.md](../rejection-sampling.md), [../../evaluation/aime.md](../../evaluation/aime.md), [../../evaluation/math500.md](../../evaluation/math500.md)

---

## What it is

Test-time training (TTT) adapts a model on the deployment distribution using only unlabeled inputs. The natural extension to reasoning — replace ground truth with **majority-vote** pseudo-labels — is fragile: one wrong vote poisons every token in every rollout. TTPO is a TTT objective for LLM reasoning that survives frequent pseudo-label errors by treating agreeing and disagreeing rollouts asymmetrically.

## How it works

For each unlabeled prompt $q$, sample $G$ rollouts $\{o_1,\dots,o_G\}$ and take the majority-vote answer $\hat{y}$.

**Two branches on the same batch:**

1. **Distill agreeing rollouts (OPSD branch).** For every $o_i$ whose final answer equals $\hat{y}$, apply an on-policy self-distillation loss toward the current policy's own high-confidence choices. **Token gating:** downweight positions where the token distribution has already converged (low remaining entropy) — no free updates.
2. **Penalize disagreeing rollouts (Grouped-RL branch).** For every $o_i$ whose final answer disagrees with $\hat{y}$, apply a grouped-RL negative advantage (GRPO-style within the same $G$ group). **Token gating:** apply only at positions where the model was *confidently wrong* — high per-token confidence with disagreeing outcome. Low-confidence tokens don't get punished for pseudo-label noise.

The empirical observation that justifies asymmetry: **rollouts disagreeing with the pseudo-label are wrong more often than the pseudo-label is wrong itself**. So the disagreeing branch is a robust "don't do this" signal even under bad votes; the agreeing branch's confidence gate blocks the pathological case of distilling into a wrong majority.

Majority-vote routing is self-improving: as the model gets better, votes get tighter, and both branches sharpen.

## Why it matters

Turns unlabeled test-time data into a training signal for reasoning without a verifier, a reward model, or human labels. Matches label-supervised OPSD on AIME / MATH-500 / AMC / Olympiad benchmarks; lifts Qwen3-1.7B from **38.0 → 45.2%** in TTT, and by **+25.2 to +36.4** in the "no-thinking" regime. Cross-task generalization holds, so the adaptation is not a narrow overfit.

## Gotchas & tricks

- **Both gates matter.** Removing token-level gating on either branch degrades quickly: the distillation branch starts reinforcing already-decided tokens, and the RL branch starts punishing hesitations that were probably right.
- **Group size $G$ controls vote reliability.** Small $G$ (e.g. 4) is too noisy; the paper uses larger groups typical of GRPO.
- **Not an alignment tool.** TTPO improves task performance under the same reward implicit in majority voting — it will happily converge to a confident but wrong consensus if the model is systematically biased on a task.

## Sources

- Paper: *TTPO: Test-Time Policy Optimization* — Wang et al., 2026 — [arXiv:2608.27448](https://arxiv.org/abs/2608.27448)
