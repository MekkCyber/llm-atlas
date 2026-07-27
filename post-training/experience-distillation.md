# Experience Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In-context learning (ICL) from an agent's own trial-and-error history is highly sample-efficient, but the gain vanishes once the context is dropped. Experience Distillation **internalizes the ICL uplift into model weights using only the trajectories already collected** — no additional environment interaction. Retains ≥64.8% of ICL gains vs 3.8% for plain SFT on the same trajectories; matches classical RL with ≥9.6× fewer environment samples.

**Prereqs:** [_post-training.md](./_post-training.md), [rejection-sampling.md](./rejection-sampling.md).
**Related:** [on-policy-distillation.md](./on-policy-distillation.md) · [_rl.md](./_rl.md) · [reopd.md](./reopd.md)

---

## What it is

The setting:

1. Agent is given a task in a new environment.
2. It performs a fixed number of *in-context* trial-and-error rollouts, using its own accumulating trajectory as context. Success rate rises across trials — this is the "ICL uplift."
3. **Question:** can we bake that uplift into the weights, so a fresh instance (no context) matches the in-context version?

Naive answer: SFT on the successful trajectories. Empirically this recovers ~4% of the ICL gain. Experience Distillation is a training objective designed to preserve the *reasoning shape* ICL exploits, not just imitate the final tokens.

## How it works

Given ICL trajectories $\tau = (s_1, a_1, \dots, s_T, a_T)$ produced with a growing context of prior attempts:

- **Extract** the pre-context (task) and the post-context (agent's most recent trajectory) from each ICL run.
- **Train** the student to produce the post-context tokens conditioned *only on the pre-context* — no attempt history in the input.
- The loss encourages the student to internalize the pattern the ICL context provided (recovery from typical mistakes, task-specific heuristics) as a weights-level prior.

Concretely, the objective differs from SFT in *what states are used as inputs*: SFT would train on the concatenated attempt history; Experience Distillation strips that history so the model must learn the pattern rather than reference it.

## Why it matters

Environment samples are the scarce resource for real-world agents — an experiment costs hours, a user's feedback costs a real human, a browser action costs a real page load. Experience Distillation turns exploration cost into a one-time expense: run ICL to learn the environment (cheap in samples, expensive in inference), then distill the learned pattern into weights (no new samples). At 9.6× fewer environment samples than classical RL for equal performance, it's the strongest published number for reusing exploration trajectories as training signal.

## Gotchas & tricks

- **ICL uplift must exist.** Experience Distillation preserves ICL gains — if the base model doesn't learn in context on this task, there's nothing to distill.
- **Not a KD replacement.** Uses no external teacher; the student's own high-quality (post-ICL-context) trajectories are the target. This makes it distinct from OPD.
- **Domain sensitivity.** Reported on software-engineering tasks (749 curated) and text adventure games. Whether the pattern transfers to open-ended web/browser agents is untested.
- **Selection matters.** Which trajectories to distill from — all runs, successful only, high-return trajectories — is a hyperparameter the paper explores; broad reuse is safer than tight filtering.

## Sources

- Paper: *Sample-Efficient Learning from Agent Experience* — Gou, Tu, Fang, Cai, Rezatofighi, 2026 — [arXiv:2607.21051](https://arxiv.org/abs/2607.21051)
