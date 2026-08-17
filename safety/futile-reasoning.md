# Futile reasoning

*Depth — the failure mode where a reasoning-tuned LLM generates long, plausible-sounding, computationally expensive derivations that arrive at incorrect answers on beyond-capability tasks.*

**TL;DR:** Reasoning-RL-tuned models are trained to produce long chain-of-thought traces because longer traces tend to correlate with correct answers on solvable problems. On problems the model *cannot* solve, that incentive still fires: the model produces long, confident, incorrect derivations that mislead users because they *look* like careful reasoning. This "futile reasoning" is a distinctive alignment failure of the reasoning-model class, not a generic hallucination. The current mitigation is [CaRL (capability-aligned RL)](../post-training/reasoning/capability-aligned-rl.md).

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [sandbagging.md](sandbagging.md), [mismatched-generalization.md](mismatched-generalization.md), [../post-training/reasoning/capability-aligned-rl.md](../post-training/reasoning/capability-aligned-rl.md)

---

## What it is

A reasoning model produces four output shapes on any task:

|  | short answer | long CoT + answer |
| --- | --- | --- |
| **correct** | correct-short | correct-long |
| **incorrect** | wrong-short | wrong-long (**futile reasoning**) |

The bottom-right cell is the safety-relevant one. Long CoT gives the wrong answer an air of *rigor* — the user sees pages of derivation, thinks the model has genuinely worked the problem, and trusts the final answer more than they would trust an unadorned wrong guess. Because reasoning-RL rewards long CoT globally (it correlates with correct on solvable problems), the model has no incentive to *stop* generating long CoT on unsolvable problems. The result: high-quality-looking fabrication on out-of-capability tasks.

Futile reasoning is not identical to hallucination:

- **Hallucination** = confidently asserting a false fact, typically in a short answer.
- **Futile reasoning** = generating a full apparent derivation whose steps are individually plausible but which arrives at a wrong answer.

The reasoning trace makes the failure both more convincing and harder to spot.

## How it works

**Why it emerges:**

1. Reasoning models are trained with a reward $r(q, o) = 1$ if the answer verifies, $0$ otherwise. There is no penalty for a wrong-long trace beyond the missed positive reward.
2. Length correlates with correctness *on the model's capability frontier* — it's near-lossless on solvable problems and neutral-to-slightly-negative on unsolvable ones.
3. Over training, the policy converges toward "always produce a long CoT ending in *some* answer" because doing so preserves the small chance of accidentally matching the verifier on hard problems.
4. There is no gradient signal telling the model "you're out of your depth here, stop."

**Diagnostics:**

- **Long-CoT wrong-rate.** Track wrong-long / (wrong-long + wrong-short). If it's much higher than the base model's ratio, the RL policy has learned to fabricate at length.
- **Length distribution by verifier outcome.** If length distributions of correct and wrong rollouts on a benchmark overlap heavily, the model isn't using length to signal uncertainty.
- **Held-out unsolvable set.** Include problems constructed to be unsolvable in the eval and watch what the model produces — refusal (good), short guess (medium), long derivation (worst).

**Mitigations:**

- [Capability-aligned RL (CaRL)](../post-training/reasoning/capability-aligned-rl.md) — reshape the reward to prefer refusal over long fabrication on out-of-capability problems; add hindsight refusal augmentation.
- Length penalties on wrong answers — modest disincentive that helps but doesn't solve the problem alone.
- Calibrated refusal prompts / classifier post-hoc — refuse based on model self-reported confidence; leaks capability info to users but doesn't fix the underlying policy.

## Why it matters

- **Highest-trust failure mode of the reasoning-model class.** A long, careful-looking wrong answer is more dangerous than a short one because users defer to it.
- **Distinct from sandbagging.** [Sandbagging](sandbagging.md) is *deliberate* under-performance. Futile reasoning is *unintentional over-attempt* — same visible failure, opposite cause.
- **RL-induced, not pretraining-induced.** The bias toward long fabrication comes from post-training, so it can be *unlearned* by post-training; you don't need to touch the base.
- **Universal to reasoning-tuned models.** Any model trained with long-CoT-RL under a purely-verifier reward inherits the incentive.

## Gotchas & tricks

- **Confusing over-refusal with fixing futile reasoning.** A model can be pushed into refusing everything hard; measure calibration, not just refusal rate.
- **Naive length penalties overshoot.** Penalizing all long CoT damages performance on solvable problems too. Penalize only when the answer is wrong (via verifier) or use conditional length shaping.
- **Refusal must be model-generated, not templated.** Templated refusals ("I don't know") can be gamed by prepending them to a fabricated derivation. Detect refusals by whether the model *actually stopped* reasoning.
- **Watch for interaction with reward hacking.** A model that learns to always refuse gets $\alpha < 1$ CaRL reward on everything — better than fabricating, worse than solving. Ensure the "solve if you can" incentive dominates.

## Sources

- Paper: *Knowing When to Quit: Diagnosing and Training LLMs to Abort Futile Reasoning* — Guan, Zeng, Xin, Lu, Lin, Han, Sun, Meng, 2026, [arXiv:2607.29211](https://arxiv.org/abs/2607.29211) — diagnoses the failure mode and introduces CaRL.
- Paper: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL* — DeepSeek-AI, 2025, arXiv 2501.12948 — the reasoning-RL recipe that instantiates the incentive.
