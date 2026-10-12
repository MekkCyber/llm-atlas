# Rubric Rewards
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reward family that complements learned preference models with **explicit rubric graders** — one per failure mode (prompt faithfulness, no unrequested additions, anti-reward-hacking patterns). Composed non-linearly with the preference reward: a naive weighted sum fails, a structured composition keeps every signal active. Produced a FLUX.2-dev variant +69 Elo over base and an Ideogram-4 variant at 1223.5 Elo on the open-source T2I Arena.

**Prereqs:** [_rewards.md](./_rewards.md), [_rl.md](./_rl.md)
**Related:** [dpo.md](./dpo.md) · [grpo.md](./grpo.md) · [cot-reward-model.md](./cot-reward-model.md)

---

## What it is

A reward design pattern: pair the dominant *preference* reward (learned from pairwise comparisons) with several *rubric* rewards, each targeting a named failure the preference model is known to miss or reward-hack. Each rubric is an explicit grader (an LLM/VLM judge or rule check) producing a bounded, interpretable score.

## How it works

- **Preference reward** $r_{\text{pref}}$: standard learned reward from human/arena pairs.
- **Rubric rewards** $r_1, \ldots, r_k$: one per failure mode. Example rubrics for T2I include "every requested object is present," "nothing was added that wasn't requested," "no watermark artefact," "no cliché over-styling." Each is a graded signal with a known contract.
- **Composition.** Not a weighted sum. The paper reports naive $\sum w_i r_i$ collapses to the dominant signal; the proposed composition is non-additive — rubrics act as *gates* and *penalties* on the preference term rather than parallel contributors. The exact form is paper-specific.

Training uses the composed scalar as the reward in standard RL post-training (PPO/GRPO-style) on the generator.

## Why it matters

Reward-only RL on preference models reliably reward-hacks — the generator exploits preference-model blind spots. Rubrics are a cheap diagnostic *and* a direct training signal for those blind spots. The result is a frontier-leaderboard T2I model trained post-hoc without re-collecting preference data. The pattern is portable: any domain with "preference is right overall but misses specific failure classes" can benefit.

## Gotchas & tricks

- Rubric grader quality is a ceiling. If the "no unrequested additions" judge is noisy, that term just injects noise.
- Non-linear composition isn't an arbitrary choice — a weighted sum lets the preference term dominate, which reinstates the hacking the rubrics were there to catch. Gating / penalising structure is the whole point.
- Rubrics must be *orthogonal* to the preference reward. A rubric that measures the same quality the preference model already covers just adds noise.
- Releases include **Arena-T2I-Training** (1K prompts); for text LLMs, the analogue would be building rubrics for instruction-skipping, style injection, hallucinated facts.

## Sources

- Paper: *Post-Training Frontier Text-to-Image Models by Composing Preference and Rubric Rewards* — Hsu et al., Arena Intelligence / UCLA, 2026 — [arXiv:2610.02967](https://arxiv.org/abs/2610.02967)
