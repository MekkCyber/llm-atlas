# Self-Improving Coding Agents (Gödel → Darwin → Mendel)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A family of agents that **rewrite their own source code** to improve on a coding benchmark. The default "clonal" version edits an agent from a single failure trajectory. **Mendel Gödel Machine (MGM)** extends this with two new mutation operators drawn from Mendelian genetics: **reaction-norm mutation** (edit using multiple task trajectories simultaneously) and **cross-lineage hybridization** (edit using a reference agent's trajectory on the same task). Both extract *comparative* signal from the archive of past attempts rather than treating each self-edit as a single-trajectory event. Introduced in *Mendel Gödel Machine* (UESTC / LMU Munich, 2026).

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

A self-improving coding agent is a program that (1) attempts coding tasks, (2) reads its own source, and (3) proposes edits to that source that should improve future performance. The Gödel-Machine framing is classical (Schmidhuber); recent revivals — Darwin-Gödel Machine and now **Mendel Gödel Machine** — use LLMs as the mutation operator and maintain an **archive** of past agent versions and their trajectories.

Two design axes:

- **What signal drives the mutation?** A single trajectory (clonal mutation) or a *comparison* across trajectories (Mendelian mutations).
- **What is the archive?** A store of past programs, or an active source of comparative gradient?

MGM's contribution is on the second axis: the archive becomes a source of comparative signal, not just a warehouse.

## How it works

MGM keeps three mutation operators:

1. **Clonal mutation (baseline).** Sample a failure trajectory. Ask the LLM: given this agent's code and this failure, propose an edit that would fix it. Standard Darwin-Gödel Machine move.

2. **Reaction-norm mutation.** Sample $k$ trajectories of the *same* agent on $k$ *different* tasks. Ask the LLM: given this agent's code and its behavior across these tasks, propose an edit that generalizes. The name comes from population genetics — the "reaction norm" is how a genotype expresses across environments. The mutation is chosen to improve the *envelope* of behavior, not just one point.

3. **Cross-lineage hybridization.** Sample a reference agent from another lineage and its trajectory on the same task. Ask the LLM: given the two agents' code and their trajectories on this task, propose an edit that transfers the reference agent's strength. Effectively "learn from a sibling that got it right (or failed differently)."

The archive is queried to construct the multi-trajectory / cross-lineage prompts. Selection of which mutations to keep uses standard evolutionary criteria (benchmark score, novelty).

## Why it matters

- **More signal per LLM call.** The dominant cost is LLM inference for the mutation step. Squeezing more signal per call — by conditioning on multiple trajectories or a sibling — beats simply running more clonal mutations.
- **Uses the archive.** Prior self-improvement work treated the archive as inert storage. MGM turns it into a comparative-signal source, which likely generalizes to any evolutionary program-synthesis setting.
- **Sample-efficient path to compound improvement.** The whole point of self-improving agents is *compound* gains. Sample-efficient mutation is the bottleneck; MGM directly attacks it.

## Gotchas & tricks

- **Mutation-prompt engineering matters.** The two new operators need multi-trajectory prompts that fit in context and stay legible to the LLM; poorly formatted prompts wipe out the comparative signal.
- **Selection pressure is a footgun.** Aggressive selection collapses lineages; too little pressure and the archive fills with near-duplicates. MGM's two new operators partially rely on lineage diversity to work — over-pruning the archive kills cross-lineage hybridization.
- **Reproducibility.** Self-improving agent scores are noisy across seeds; report multiple runs per configuration.
- **The "Gödel" is aspirational.** Formal self-referential improvement guarantees do not hold; these are LLM-driven heuristic self-edits.

## Sources

- Paper: *Mendel Gödel Machine: Recursive Self-Improving Coding Agents via Comparative Evolution* — Liu, Liu, Yan, Tresp, Ma (UESTC / LMU Munich / MCML), arXiv 2608.07645, 2026.
- Earlier: Darwin-Gödel Machine and the broader self-improving-agent line (see paper for full lineage).
