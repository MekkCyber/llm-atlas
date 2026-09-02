# Rubric-as-Reward
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RL post-training for open-ended agent tasks (where a rule-based verifier doesn't exist) that uses a **rubric** — a structured, section-derived evaluation criteria list — as both the reward signal *and* as privileged context during SFT. Introduced at scale in PaperGym for research-plan generation, where the rubric is derived from paper *methods+experiments* while the question comes from *goals+background*, driving **criterion leakage** from 12–34% (prior datasets) down to **3.7%**.

**Prereqs:** [rlvr.md](rlvr.md), [_rewards.md](_rewards.md)
**Related:** [grpo.md](grpo.md) · [cot-reward-model.md](cot-reward-model.md) · [reasoning/prm.md](reasoning/prm.md) · [../data/decontamination.md](../data/decontamination.md)

---

## What it is

RLVR ([rlvr.md](rlvr.md)) trains on tasks where a cheap verifier (math answer match, unit tests) can score outputs. It fails when the task is open-ended — "propose a research plan," "write a design doc," "critique this experimental setup" — because no such verifier exists. Rubric-as-reward fills that gap: a set of concrete criteria derived from a *source of truth* (a paper, a solution, a reference plan) is used to grade candidate outputs, giving a scalar (or per-criterion) reward that can drive GRPO / PPO.

The key discipline is that **the rubric must not leak the answer**. If the rubric is derived from the same content that constitutes the target output, the policy learns to reverse-engineer the rubric rather than to reason.

## How it works

**Two-stage training (as in PaperGym):**

1. **SFT with rubric as privileged context.** Model sees the rubric alongside the prompt during supervised fine-tuning. Learns *how to satisfy* rubric criteria explicitly.
2. **RL with rubric as reward.** Rubric is now hidden. Model generates freely; each output is scored against the rubric (either scalar via LLM judge or per-criterion), and the score drives GRPO updates. Because SFT taught the model what the rubric looks for, RL sharpens its ability to satisfy it without seeing it.

**Leakage discipline.** PaperGym's core trick: derive the **question** from the paper's *goals + background* sections; derive the **rubric** from *methods + experiments*. The two sources overlap semantically but share almost no surface content, so the question doesn't directly imply the rubric. Result: **3.7%** criterion leakage vs 12–34% in prior similarly-shaped datasets.

**Reward shape.**
- Pass/fail per criterion → sum of indicators (simplest).
- Continuous per-criterion (via LLM judge) → mean or weighted sum.
- Weighted by criterion importance → requires curator input.

## Why it matters

- **Extends RLVR to open-ended tasks** without hand-writing a verifier.
- **Rubric-derived rewards are auditable.** Unlike an opaque reward model, each criterion is a text string; failure modes are inspectable.
- **Concrete evidence of scale advantage.** In PaperGym, Qwen3-8B trained with rubric-as-reward reaches **73.48 on ResearchQA**, beating the much larger Kimi K2.6 baseline — direct evidence that shaped rewards over general capabilities can transfer.
- **Cheap to build.** For any curated corpus (papers, code reviews, medical case notes), you can derive question/rubric pairs mechanically and start training.

## Gotchas & tricks

- **Rubric leakage is the whole ball game.** If your question already implies the rubric, RL just learns to parrot criteria. Test on a held-out set: does the model win on the trained rubric but flunk paraphrased versions?
- **Judge model is a reward model in disguise.** All reward-hacking pathologies of LLM judges still apply — verbosity bias, position bias, sycophancy. Ensemble judges or ratio-check with a stronger judge.
- **Two-stage matters.** Trying to do rubric-as-reward without the rubric-as-privileged-context SFT stage often fails: the base model doesn't know what rubric-satisfaction looks like and gradients are too sparse.
- **Rubric quality bounds policy quality.** A model can only learn what the rubric measures. Vague criteria produce vague policies.
- **Interaction with reward hacking.** LLM judges reward polished-sounding-but-empty outputs at very high rates. Include structural or content-anchored criteria to counterweight.

## Sources

- Paper: *PaperGym: Rubric-Centered Evolution for Research-Plan Generation* — Wang, Lu, Yan, Song, Zhang, Lu, Xiao, Zhuang, Shen — Zhejiang U., 2026 — arxiv.org/abs/2608.31119.
