# Rubric Rewards
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reward family that scores a generation against *per-prompt rubrics* — checklists derived from the prompt itself — rather than against generic preference. When combined with a classic preference reward (Bradley–Terry over human votes), rubrics catch prompt-faithfulness, intent-dependent constraints, and common reward-hacking patterns. A naïve weighted sum of the two signals optimizes poorly; the composition strategy introduced by Arena Intelligence (2026) balances them explicitly. On text-to-image, applied to FLUX.2-dev → **+69 Elo** on the Arena T2I leaderboard; applied to Ideogram-4 → **1223.5 Elo**, the top open-source entry.

**Prereqs:** [_rewards.md](_rewards.md), [dpo.md](dpo.md), [grpo.md](grpo.md)
**Related:** [cot-reward-model.md](cot-reward-model.md) · [_rl.md](_rl.md)

---

## What it is

Preference reward models answer "which of these two outputs is better in general." Rubrics answer "did the output do what this specific prompt literally asked for." These signals are *orthogonal*: a model can win on preference by producing beautiful outputs that ignore the prompt; it can win on rubrics by satisfying constraints while being bland. Rubric rewards exist to capture the second axis.

## How it works

- **Preference reward.** A Bradley–Terry head trained on large-scale human votes between pairs of outputs for the same prompt.
- **Rubric construction.** For each prompt, derive a short checklist of intent-dependent constraints (object counts, color/geometry constraints, text content, style instructions). Rubrics can be auto-generated from the prompt by an LLM or supplied by annotators.
- **Rubric reward.** Score each constraint 0/1 (or a small graded scale) using an automated verifier (VLM for images, parser for structured outputs). Aggregate to a scalar.
- **Composition strategy.** Instead of $R = \alpha R_\text{pref} + (1-\alpha) R_\text{rubric}$, the paper's composition explicitly balances the two to avoid gradient collisions between an aesthetic preference signal and a satisfaction constraint signal.
- **Anti-hacking rubrics.** A subset of rubrics specifically target known reward-hacking modes (text overlays, duplicate subjects, inpainted artifacts) so the policy cannot escape the preference RM through them.

## Why it matters

- Gives open-source post-training a recipe to catch up with closed-source models whose main lever has been post-training over preference signals.
- Separates "what humans like on average" (preference RM) from "what this prompt asked for" (rubrics), which was collapsing into a single hackable scalar before.
- The composition strategy is explicit about the preference-vs-constraint tradeoff and avoids naïve-weighted-sum failure modes.

## Gotchas & tricks

- Auto-generated rubrics inherit the generator's priors; cross-check with human rubrics on a validation slice before using at scale.
- Rubric graders are themselves models (VLMs, parsers); their errors become reward noise and have to be budgeted for.
- Anti-hacking rubrics are a treadmill: once the policy stops hacking the preference RM in a known way, it will find new ways.

## Sources

- Paper: *Post-Training Frontier Text-to-Image Models by Composing Preference and Rubric Rewards* — Hsu, Angelopoulos, Chiang, Stoica, Hsieh et al., Arena Intelligence / UCLA, 2026 — [arXiv:2610.02967](https://arxiv.org/abs/2610.02967).
