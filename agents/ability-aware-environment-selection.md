# Ability-Aware Environment Selection (AES)
*Depth — diversity selection over environment pools for agent RL.*

**TL;DR:** Naively pooling training environments for agents produces heavy duplication along ability axes — many environments testing the same skill, few testing others. AES scores candidate environments by the *ability signature* they exercise (perception, planning, tool-use, memory, ...) and selects a subset with balanced coverage, replacing "collect as many environments as possible" with "collect environments that diversify the ability profile."

**Prereqs:** [README.md](README.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [environment-curriculum.md](environment-curriculum.md), [../data/_data-curation.md](../data/_data-curation.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

Agent RL pools accumulate environments faster than practitioners can characterize them. AES adds an explicit selection step: score each candidate environment by which abilities it demands, then pick a subset that spreads coverage across the ability space, rather than one that piles on the most common ability profile.

## How it works

1. **Ability inventory.** Define a small set of ability dimensions (e.g. perception, low-level control, planning, tool-use, memory, instruction-following). This is manual; the paper picks a fixed inventory.
2. **Per-environment ability score.** For each environment $E_j$, compute a vector $a_j \in \mathbb{R}^k$ giving the fraction of trajectories in that environment that exercise each ability. Scoring can be model-elicited (a strong reference model rates the environment) or trajectory-derived (e.g. tool-call frequency, perception-required frames).
3. **Diversity selection.** Select a subset $S \subseteq \{E_j\}$ under a budget that maximizes coverage — greedy submodular selection over ability vectors is a natural choice. Reject candidates whose ability vector is already saturated in $S$.

The result is an environment pool with balanced coverage across abilities, versus the natural "long-tailed" distribution you get from ingesting environments raw.

## Why it matters

- **Fixes the environment-scaling plateau from the other side.** Pairs with HDC ([environment-curriculum.md](environment-curriculum.md)) — HDC schedules difficulty, AES decides which environments enter the pool at all.
- **Same lesson as data curation.** Naive scaling assumed all environments were roughly equal — the same fallacy that "more tokens = more capability" was for pretraining before Chinchilla.
- **Cheap to apply.** Scoring is a one-time preprocessing pass; selection is a bounded combinatorial problem.

## Gotchas & tricks

- **Ability inventory is the whole game.** The categories you pick determine what "diverse" means. A too-narrow inventory selects a homogeneous pool that just spans your inventory.
- **Model-elicited scores can drift.** If a stronger reference model reveals new abilities, prior selections may look poorly balanced retroactively.
- **Not a substitute for curriculum.** AES gives you *what* to train on; curriculum ([environment-curriculum.md](environment-curriculum.md)) gives you *in what order*. Both matter.
- **Diminishing returns at fixed budget.** Once every ability is well-covered, further diversity work is wasted; return to difficulty scaling.

## Sources

- Paper: *Beyond Simply Environment Scaling: Designing Effective Environment Distributions for Multimodal Agent Learning* — Zhu et al., CAS/UCAS, 2026 — arXiv:2608.03571.
