# RSTG: Recovering Signals via Adaptive Teacher Guidance

*Depth — a surgical way to combine GRPO with On-Policy Distillation that rescues zero-variance groups without killing exploration.*

**TL;DR:** GRPO loses all gradient when every response in a group gets the same reward. Naively stacking On-Policy Distillation (OPD) on top rescues signal but breaks in three ways: not all samples benefit from distillation, fast-fitting to the teacher kills exploration, and OPD's advantages are asymmetric. RSTG (Han et al., 2026) applies OPD **only** to negative zero-variance prompts, weights each sample by teacher confidence, targets only tokens with high student entropy or large teacher-student divergence, and adds SFT on teacher-correct trajectories for extra positive gradient. **+4.02 pp math, +3.05 pp code** over the naive GRPO + OPD baseline.

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md), [on-policy-distillation.md](on-policy-distillation.md)
**Related:** [sa-opd.md](sa-opd.md) · [rejection-sampling.md](rejection-sampling.md) · [_post-training.md](_post-training.md)

---

## What it is

Three interacting failure modes of naive GRPO + OPD:

- **Sample-level:** OPD applied to prompts where GRPO already has good signal wastes teacher compute and dilutes the RL gradient.
- **Fitting-speed:** OPD's dense signal makes the student converge toward the teacher quickly, suppressing the exploration that made RL worth doing.
- **Advantage asymmetry:** OPD's forward-KL update tends to suppress most tokens and boost a few — pushing the policy peaked rather than better-calibrated.

RSTG addresses all three with a three-part restriction.

## How it works

1. **Sample-level restriction.** For each GRPO prompt, check the group's reward variance. If all responses got the same reward *and* the reward is negative (they all failed), apply OPD to that prompt; otherwise skip. The OPD signal is added to precisely the prompts where GRPO produced *no* gradient.
2. **Sample weighting.** Each such sample is weighted by the teacher's confidence score on the correct trajectory. Low teacher confidence → low weight (don't inject noisy signal).
3. **Token-level restriction.** Within an OPD-eligible sample, only distill tokens where the student's entropy is high (the student is uncertain) *or* where teacher-student divergence is large. Otherwise skip the token.
4. **Positive-signal SFT.** Additionally SFT on teacher-generated correct trajectories, injecting positive gradient where RL and OPD both provide none.

The full recipe is: GRPO (as usual, works on non-zero-variance prompts) + gated OPD (on negative-zero-variance prompts, filtered tokens, teacher-weighted) + SFT on teacher-correct rollouts.

## Why it matters

- **Solves the most-cited operational failure of RLVR.** Zero-variance groups are the primary reason production GRPO pipelines lose training efficiency; RSTG is the cleanest published fix.
- **Preserves exploration.** Because OPD is skipped on any prompt with non-zero reward variance, the RL exploration loop stays intact on prompts where it's working.
- **General recipe.** Not tied to a specific benchmark or teacher — pattern applies to any RLVR + teacher-access setting.

## Gotchas & tricks

- **Teacher confidence scoring must be calibrated.** If the teacher is overconfident on wrong answers, RSTG will up-weight those and teach the student to be wrong-with-confidence.
- **The three restrictions compose multiplicatively.** In production runs, you may find *very few* tokens actually receive OPD gradient — this is by design, but check that the OPD contribution is non-trivial before spending teacher compute.
- **Stacks with SA-OPD.** RSTG picks which prompts and roughly which tokens; [sa-opd.md](sa-opd.md) provides a finer input-groundedness filter on the surviving tokens.
- **Positive-only SFT ≠ rejection-sampling SFT.** RSTG's SFT step uses teacher-generated correct trajectories, not student self-generated ones — closer to teacher distillation than to [rejection-sampling.md](rejection-sampling.md).

## Sources

- Paper: *Distill Where You Fail: Recovering Learning Signals of Negative RL-Groups from Adaptive Teacher Guidance* — Han, Xiao, Lu, Jin, Yao, Liu, Hao, Sun, Yang, Gu, Cai, Xiong, 2026 — [arXiv 2608.00782](https://arxiv.org/abs/2608.00782). TJUNLP Lab (Tianjin U.) + Meituan Longcat.
