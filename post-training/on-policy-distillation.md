# On-Policy Distillation for Long-Horizon Agents (CARE-OPD)
*Depth — student-seeded teacher completions, confidence-gated response selection, and hindsight replay allocation for cheap on-policy distillation into long-horizon LLM agents.*

**TL;DR:** Fully online on-policy distillation (OPD) is expensive: every training update needs fresh environment rollouts, which dominate cost for long-horizon agents (SWE-bench, Deep Research). **Prefix replay** cheapens this by caching trajectory prefixes and resampling student completions from them, but has two weaknesses: teacher-only prefixes miss histories the student would actually reach, and fixed position-based schedules under-weight late-trajectory decisions. **CARE-OPD** fixes both with (a) *student-seeded teacher completions*, (b) *response-level confidence gating*, and (c) *hindsight replay allocation* that uses downstream execution evidence to re-prioritize which prefixes to replay.

**Prereqs:** [rejection-sampling](rejection-sampling.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)
**Related:** [splice](splice.md) · [rl-prompt-curation](rl-prompt-curation.md) · [grpo](grpo.md) · [../agents/README.md](../agents/README.md)

---

## What it is

On-policy distillation asks a strong **teacher** to produce target behavior for the **student**'s own on-policy states — so the student learns to imitate the teacher *on the trajectories the student actually visits*, not on the teacher's own preferred trajectories. For long-horizon agents this is expensive because fresh rollouts dominate cost.

Prefix replay caches trajectory prefixes and generates fresh student completions from them, amortizing rollout cost across many SFT updates. Its two failure modes:

- **Prefix bias.** If prefixes come only from the teacher, they miss the histories the student would actually reach. The student then learns to imitate from states it will never see.
- **Flat allocation.** Position-based replay schedules (e.g. "replay each prefix from step $t$ with uniform probability") miss that some decisions matter more than others, especially late-trajectory decisions whose downstream outcomes can only be evaluated after the fact.

CARE-OPD addresses each directly.

## How it works

Three pieces, applied together:

1. **Student-seeded teacher completions.** Instead of collecting prefixes from teacher rollouts, start from *student* rollouts up to step $t$, then ask the teacher to complete from there. The replay buffer is now indexed by student-visited states.
2. **Response-level confidence gating.** Each teacher completion is scored by the teacher's own confidence (e.g. average log-prob, or a lightweight check). Low-confidence completions are dropped before they enter the training batch. This prevents the student from being pulled toward noisy teacher behavior.
3. **Hindsight replay allocation.** After a batch of trajectories terminates and receives environment reward, re-weight the prefixes by *downstream execution evidence*. Prefixes whose successor steps actually mattered for the outcome get replayed more; prefixes that led to generic no-ops get dropped. This is a curriculum signal that the student discovers *from the trajectory's future*, not from its position.

Steps 1 and 2 produce a *targeted* replay buffer; step 3 *allocates compute* within it.

## Why it matters

- **Same accuracy at lower cost.** On SWE-bench Verified, CARE-OPD reports **59.5% task success** with **~45% less end-to-end training time and 44% fewer environment actions** than asynchronous online OPD.
- **Student-seeded replay is the key correctness fix.** Teacher-seeded prefix replay in prior work was consistently shown to underperform online OPD on long-horizon agents; CARE-OPD closes that gap without giving up prefix replay's cost advantage.
- **Hindsight allocation generalizes.** The "use downstream execution evidence to decide where to spend the next rollout" pattern transfers to RL curricula ([rl-prompt-curation](rl-prompt-curation.md)) and to replay-buffer design in any long-horizon RL setup.

## Gotchas & tricks

- **Confidence gating needs calibration.** A teacher that is overconfident on OOD prefixes will let bad completions through; a teacher that is underconfident will reject usable ones. The paper uses a response-level confidence check — per-completion, not per-token — which is more robust than token-level log-probs for long sequences.
- **Student-seeded prefixes drift with training.** Each epoch the student's policy changes, so old prefixes become mismatched. Periodically refresh the prefix buffer; the paper refreshes at a fixed cadence.
- **Hindsight needs cheap counterfactuals.** To know which step "mattered," you need either an environment that supports partial replay or a learned critic — a lightweight per-step critic works and is cheaper than full counterfactual rollouts.
- **Doesn't eliminate online rollouts, reduces them.** CARE-OPD still needs *some* fresh rollouts to seed the prefix buffer and to compute hindsight signals. The gain is a 2× reduction, not a 10× one.

## Sources

- Paper: *Collect for Coverage, Replay with Evidence: Adaptive and Efficient On-Policy Distillation for Long-Horizon Agents* — Li, Li, Liu, Huang, Dong, Wang, Tuo, Wu, Peng, Tian, Zhao, Wang, Wang (SJTU / SII / JD.com), 2026, arXiv 2611.34756.
- Related: [splice](splice.md) — companion recipe for agentic RL cost reduction, different lever (all-fail GRPO groups instead of OPD).
- Related: [../systems/partial-rollouts.md](../systems/partial-rollouts.md) — prefix replay is a partial-rollout pattern.
