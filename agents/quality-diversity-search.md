# Quality-Diversity Search for LLM Ideation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Cast LLM-driven research ideation as **Quality-Diversity (QD) search** — a mature framework from evolutionary computing where a population is maintained across a *behavior space* rather than optimized for one objective. Ideas evolve in lineages, quality is driven by multi-objective feedback (repair + refinement), and diversity is driven by explicit comparison against completed / historical / rejected pools. Introduced in IDEAgent (2026); paired with a new "Yield" metric jointly measuring quality and diversity.

**Prereqs:** none
**Related:** none yet

---

## What it is

A multi-agent framework that swaps single-objective idea generation for QD-style **behavior-space coverage**. Instead of "the best idea," the objective is *coverage of the space of plausibly-good ideas*. Directly addresses the failure mode of open-ended LLM ideation systems that collapse to a handful of clustered proposals.

## How it works

- **Lineages.** Each idea has a lineage — its parent (what it was refined from) and its siblings (what alternatives were considered). Lineages are the unit of evolution.
- **Quality driver.** Multi-objective feedback (criticism against multiple criteria) triggers repair and refinement inside a lineage.
- **Diversity driver.** Before accepting a new idea, compare it against three memory pools: completed (things we've already produced), historical (things we know about), rejected (things we've decided against). Reject ideas that don't move the population into new behavior-space cells.
- **Yield metric.** A joint metric combining quality and diversity into one number — 3.89× baseline Yield across 32 CS ideation topics, with non-zero Yield on 8× more topics than the best single-objective baseline.

## Why it matters

"Auto-scientist" and research-agent systems have a known collapse mode: they produce a handful of similar ideas and stop exploring. QD is the standard fix for this in evolutionary computing; grafting it onto LLM ideation is a principled remedy rather than another prompt-engineering hack. The Yield metric gives the sub-field a shared quantitative bar.

## Gotchas & tricks

- **Behavior-space definition is where the biases live.** The three-pool comparison implicitly defines the behavior space; a poorly chosen definition can force diversity along irrelevant axes.
- **Rejection-pool contamination.** Rejected ideas that were "close to accepted" can starve neighboring lineages if the diversity check is too strict.
- **Yield is topic-normalized.** Report per-topic breakdowns; a headline number can hide dramatic per-topic variance.

## Sources

- Paper: *IDEAgent: Agentic Quality-Diversity Search for Research Idea Generation* — Poria et al. (DeCLaRe Lab, NTU), 2026 — [arXiv:2607.22375](https://arxiv.org/abs/2607.22375).
