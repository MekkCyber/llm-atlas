# Harness-Augmented Agent Training (Harness-S)
*Depth — use a pool of sibling harnesses as a training-time augmentation; rescue base failures from sibling successes; route at inference.*

**TL;DR:** An "agent harness" is the prompting + scaffolding around an LLM policy (system prompt, tool docs, output format, memory structure). Harness edits are notoriously brittle: a small change swings scores non-monotonically. **Harness-S** (Jiao et al., 2026) exploits this by (1) defining a **pool of sibling harnesses**, each a small structural edit of a base harness, (2) collecting trajectories from *sibling* harnesses on tasks where the base harness fails, distilling only the successful trajectories back into a single policy, and (3) training an **adaptive router** that selects the harness at inference time. Reported gains on AppWorld, SWE-Gym and DBBench beyond policy-only training; further gains from Rank-1 routing.

**Prereqs:** [rejection-sampling](../post-training/rejection-sampling.md), [rlvr](../post-training/rlvr.md)
**Related:** [agent-memory](./agent-memory.md), [latent-context-compression](./latent-context-compression.md)

---

## What it is

Three empirical properties of harness edits motivate the method:

- **Locality:** a given edit mostly affects a small cluster of tasks.
- **Complementarity:** different edits help different clusters; their combined coverage is strictly larger than any single harness.
- **Nonstationarity:** the best harness for a task can change over training.

Harness-S treats the harness itself as an *augmentation dimension*, analogous to data augmentation. Instead of picking one harness and overfitting the policy to it, you train one policy that is competent under many harnesses and pick the right harness at test time.

## How it works

### Fixed pool of sibling harnesses

A base harness $H_0$ (production prompt + scaffold) and $N$ **siblings** $H_1, \ldots, H_N$, each a small structural edit (reorder tool docs, change output format, add/remove a memory slot). $N = 5$–$10$ typical. The pool is fixed before training.

### Rescue training

For each training task:

1. Roll out under $H_0$. If any rollout succeeds, do normal RL / SFT on that trajectory.
2. If *all* $H_0$ rollouts fail: roll out under each $H_i \ne H_0$. For sibling trajectories that succeed, transform them to look as if they were produced under $H_0$ (reconstructing tool calls, memory ops, outputs to fit the base-harness surface) and add them to the policy's training set.
3. Train the policy on the combined set under $H_0$. The policy learns to produce "base-harness-shaped" outputs that solve tasks only siblings could solve.

This is a specialization of [rejection-sampling](../post-training/rejection-sampling.md) over harness augmentations.

### Adaptive router (inference)

At inference time:

- Compute a **behavior profile** for the current task (first-few-step signatures, task embedding).
- Score each $H_i$ by (a) matching the behavior profile to training-time profiles that $H_i$ was most-successful on and (b) **LLM voting** (an auxiliary LLM reads the task + first step and ranks harnesses).
- Route the task to the top-ranked $H_i$. "Rank-1 routing" = pick only the top; soft variants weight the top $k$.

Routing gains come on top of training-time rescue — the two levers are complementary.

## Why it matters

- **Reframes prompt engineering as a training signal.** "Harness tuning" and "policy training" were historically separate. Harness-S shows they are the same problem viewed from two sides.
- **Covers a specific credit-assignment gap.** Long-horizon agent RL has thin reward signal; harness augmentation adds variance to the training distribution without touching the reward function.
- **Fits existing agent stacks.** The base harness stays intact; sibling harnesses only exist during training and routing; no downstream API changes.

## Gotchas & tricks

- **Sibling design is the limiting factor.** Too-similar siblings give no new coverage; too-different siblings make trajectory transformation lossy. Hand-designed siblings win over random edits.
- **Trajectory transformation is lossy.** Rewriting a sibling trajectory to look like it was produced under $H_0$ can drop information. Keep transforms shallow.
- **Router can over-fit profile matching.** If the training set is small, profile matching over-fits; LLM voting is a regularizer.
- **Rescue training can mode-collapse.** The policy learns to always imitate the best-performing sibling's strategy, losing diversity on tasks the base harness already solved. Mitigation: oversample $H_0$-success trajectories to maintain baseline competence.
- **Not a substitute for a good base harness.** The method lifts the ceiling; it doesn't fix a broken floor.

## Sources

- Paper: *Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training* — Jiao et al., Xiaohongshu / SJTU / CMU, 2026 — [arXiv:2611.04277](https://arxiv.org/abs/2611.04277).
- Background: *AppWorld*, *SWE-Gym*, *DBBench* — the three agent-RL benchmarks used.
- Background: [rejection-sampling](../post-training/rejection-sampling.md) — rescue training is harness-indexed rejection sampling.
