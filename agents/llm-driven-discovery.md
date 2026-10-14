# LLM-Driven Discovery (Parallel-Init Harnesses)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** OpenEvolve-style discovery harnesses iteratively mutate candidate solutions with an LLM, scoring each against a verifier. The paper audits 12 such harnesses on 5 tasks and finds success is fragile, mode-collapses are common, and most proposed diversity interventions give inconsistent gains. The actionable finding: a dedicated initial *parallel-exploration* stage (independent, wide sampling) before the iterative loop gives consistent improvements across harnesses and tasks.

**Prereqs:** [agents/README](../agents/README.md)
**Related:** [rejection-sampling](../post-training/rejection-sampling.md)

---

## What it is

LLM-driven discovery is the search paradigm where an LLM proposes code / solutions / designs, a verifier scores them, and successful candidates seed the next proposal round. Examples: FunSearch, OpenEvolve, autonomous research-code agents. The paper is less about proposing a new harness than about *characterizing* the whole class and identifying initialization as the dominant lever.

## How it works

Standard LLM-driven discovery loop:

1. Keep a *population* of candidate solutions.
2. Sample one or more candidates (mutation parents).
3. Prompt the LLM to produce a variant (mutation / combination).
4. Score with a verifier (unit tests, benchmark, custom reward).
5. Replace / expand the population.
6. Repeat.

The paper's two findings:

- **Mode collapse.** The population loses diversity over iterations; later rounds explore a narrow neighborhood of a few early winners.
- **Harness fragility.** Across 12 Modular harness variants (different mutation prompts, population sizes, selection policies) and 5 discovery tasks, overall success is highly harness-dependent.

The fix:

- **Parallel-exploration init.** Before iterative refinement, generate a *wide independent sample* of candidates — different seeds, different prompts, no shared context. This seeds the iterative loop with a diverse initial population, counteracting downstream collapse.

Reported: consistent gains across many harnesses and tasks after adding the init stage, outperforming diversity-only interventions that act inside the loop.

## Why it matters

- **Diagnoses a known pain.** Practitioners of discovery harnesses see mode collapse; this is a rigorous characterization.
- **Cheap lever.** The init stage adds parallel-sampling cost up-front, no change to the iterative loop logic.
- **Explains intra-loop interventions.** Why diversity bonuses inside the loop give inconsistent gains: they fight against the collapse once it has already biased the trajectory, instead of preventing it.

Reframe: in LLM-driven discovery, the *initial population* often determines the outcome, not the search policy.

## Gotchas & tricks

- Independence matters — if the parallel samples share prompt context too heavily, you get a wide fan of near-duplicates and the init stage adds nothing.
- Compute split between init and loop is an open hyper-parameter; the paper does not prescribe one.
- The 5-task eval is small; the generality of the init prescription at larger task breadths is still open.
- Does not substitute for a verifier — bad verifier → broad-but-wrong initial population → waste.

## Sources

- Paper: *Initialization Improves LLM-Driven Discovery* — Sakarvadia, Ciccone, Raffel — U. Chicago / Vector / U. Toronto, 2026 — [arXiv:2610.00707](https://arxiv.org/abs/2610.00707)
