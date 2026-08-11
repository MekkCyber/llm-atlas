# Persona synthesis
*Depth — dependency-graph generation of large-scale persona records with correlated attributes.*

**TL;DR:** Generating synthetic personas for user simulation typically fails in one of two ways: sampling independently over attributes produces incoherent profiles ("cost-conscious retiree who prefers frontier models"), while templating loses diversity. The MatrAIx approach samples from a **dependency graph** over 1,290 categorical attribute dimensions, preserving inter-attribute correlations; combined with a human-grounded arm (599,847 personas extracted from real profiles), the pipeline yields a 1M-persona coreset of the 8.3-billion-record Persona 8B bank.

**Prereqs:** [_data-curation.md](_data-curation.md), [README.md](README.md)
**Related:** [quality-filtering.md](quality-filtering.md), [../evaluation/persona-simulated-users.md](../evaluation/persona-simulated-users.md)

---

## What it is

A persona is a structured description of a simulated user: age, occupation, income, tech-comfort, brand affinities, geography, prior product experience, and so on. Persona synthesis is the pipeline that produces personas at scale — the raw substrate for simulated-user evaluation of AI products.

The MatrAIx approach solves the coherence problem by explicitly modeling **which attributes constrain which other attributes**, then sampling along the dependency graph rather than as independent categoricals.

## How it works

**Schema.** A fixed set of $\sim 1{,}290$ categorical attribute dimensions covers demographics, technographics, preferences, behavioral tendencies, and past-experience descriptors.

**Dependency graph.** A DAG over the attributes encodes correlations authored (or learned) from real profiles. Example edges:
- `age → income` (age constrains income range).
- `income + region → brand affinities` (joint constraint).
- `tech comfort → AI product experience`.

**Sampling.**
1. Topologically sort the DAG.
2. For each persona, sample attributes in DAG order; each conditional draw is from the empirical distribution given already-sampled parents.
3. Reject / resample samples that violate global constraints (e.g. inconsistent occupation-education combinations).

**Two-arm generation.**
- **Human-grounded arm.** 599,847 personas derived from real human-authored profiles (survey responses, published bios), converted into the schema.
- **Synthetic arm.** 400,000 personas sampled from the dependency-graph generator, quality-filtered.
- Combined into a 1M-persona released coreset from the full 8.3-billion-record Persona 8B bank.

**Persona adherence validation.** A 400-trial controlled study evaluated persona adherence across ten behavioral attributes; the declared behavior was expressed or correctly suppressed in **91.5% of trials** (366/400).

## Why it matters

- **Fixes the incoherent-persona problem.** Dependency-graph sampling preserves attribute correlations that independent categorical sampling destroys.
- **Two-arm design captures both realism and scale.** Human-grounded arm anchors realism; synthetic arm scales cheaply.
- **Enables population-scale simulated-user evaluation.** Ties directly into product-evaluation pipelines (see [../evaluation/persona-simulated-users.md](../evaluation/persona-simulated-users.md)).
- **Reusable substrate.** The released coreset can seed evaluation for any AI product, not just MatrAIx.

## Gotchas & tricks

- **DAG authorship is the whole game.** A poorly-authored graph gives you back the same incoherent personas independent sampling would.
- **Distribution shifts drift the graph.** Empirical conditionals from 2024 human profiles may not apply to 2030 populations; date-stamp the graph.
- **Quality filtering is essential for the synthetic arm.** Rejection rates on constraint violations can be high; monitor.
- **Attribute count is a lever.** More dimensions = finer distinctions but sparser conditional distributions; 1,290 is what MatrAIx picked.
- **Adherence ≠ realism.** 91.5% adherence means personas *express* their declared behavior; whether those behaviors match real humans is a separate validation.

## Sources

- Paper: *Simulating the World with 8.3 Billion Persona Agents (MatrAIx)* — Li, Hao et al. (39-institution consortium), 2026 — arXiv:2608.04205.
