# Pretraining-seed-driven jailbreak variance
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Identical safety post-training applied to sibling base models (same recipe, different pretraining seeds) produces **noticeably different jailbreak profiles**. On 1B models across 25 data recipes × 3 seeds, Cerulean Labs (2026) found refusal patterns correlate at 0.984 within a run, 0.811 across seeds, and 0.779 across data recipes — so **pretraining randomness, not recipe, dominates safety eval outcomes**.

**Prereqs:** [_jailbreaks](_jailbreaks.md)
**Related:** [_attacks](_attacks.md), [competing-objectives](competing-objectives.md)

---

## What it is

Safety evaluation usually treats a model family's safety as a property of the family: "Llama-3-8B refuses 93% of HarmBench." This paper tests that assumption with a controlled factorial: fix post-training exactly, vary the pretraining seed and data recipe independently, measure downstream refusal patterns.

The finding reframes what a safety number *is*. It is specific to the pretraining run, not the recipe.

## The experiment

- **Base models:** 1B parameters, 25 pretraining data recipes.
- **Seeds:** 3 pretraining seeds per recipe → 75 base checkpoints.
- **Post-training:** identical safety post-training (SFT + preference optimisation) applied to all 75.
- **Evaluation:** per-prompt refusal / jailbreak success across a wide attack suite.

## The results

Three correlations of per-prompt refusal patterns:

- **0.984** between two post-trained copies of the **same** pretraining run (near-perfect).
- **0.811** across **different seeds** of the **same recipe**.
- **0.779** across **different recipes**.

The gap between 0.984 and 0.811 is the pretraining-seed effect. The small additional gap between 0.811 and 0.779 is the recipe effect — comparatively minor.

**Consequence for attack transfer:** the best jailbreak against one checkpoint remains the best against a sibling checkpoint (same recipe, new seed) only **27%** of the time. Attack discovery on one seed does not transfer well to another.

**Consequence for model identification:** per-prompt refusal patterns are so stable within a run that they identify the pretraining run among 75 with high accuracy — a kind of behavioural fingerprint.

## Why it matters

- Safety evals that run on one checkpoint undercount family-level variance. Reporting a single refusal number is misleading.
- Red-team findings against one seed should be treated as preliminary until replicated on sibling seeds.
- The "pretraining seed → safety" link suggests that safety emerges partly from accidents of initialisation, not only from data and post-training recipe. The implication for alignment is uncomfortable: reproducibility of safety properties is lower than the field assumes.

## Gotchas & tricks

- **Scale.** The study is at 1B. Whether the seed effect shrinks or grows at 100B+ is open.
- **Post-training identity.** The study holds post-training *exactly* fixed. Tiny differences in post-training data order would introduce another variance source.
- **Metric choice.** Per-prompt refusal correlation is one metric; aggregate ASR correlations would be higher because errors cancel.

## Sources

- Paper: Identically Trained Language Models Break on Different Jailbreaks — Chang et al., 2026 — [arXiv:2610.34074](https://arxiv.org/abs/2610.34074)
