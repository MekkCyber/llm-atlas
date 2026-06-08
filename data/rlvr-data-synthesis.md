# RLVR Data Synthesis (Atomic Decomposition + Recombination)
*Depth — manufacturing fresh verifiable RLVR prompts at the model's frontier of competence.*

**TL;DR:** Reinforcement Learning with Verifiable Rewards is bottlenecked not by the algorithm or the verifier, but by the **supply of sufficiently hard verifiable prompts**. Heuristic seed-expansions plateau quickly. ADR (Zheng et al., 2026) decomposes seed problems into **atomic units** — constraint types, data structures, algorithmic motifs — and recombines them under a target difficulty. The verifier and the unit tests are auto-derived from the recipe, so the resulting prompts are RLVR-ready with no human labels. Scales code RLVR past the Codeforces-class ceiling.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [_data-curation](_data-curation.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [../post-training/grpo.md](../post-training/grpo.md), [../evaluation/livecodebench.md](../evaluation/livecodebench.md)

---

## What it is

A *generation* (not curation) recipe for verifiable code/math/tool-use prompts. Output: a stream of new problems plus their verifiers, drawn from a controllable difficulty distribution near the current policy's frontier.

## How it works

```
1. Seed parse:
   For each seed problem in a public corpus (LeetCode-style, math, etc.):
     decompose into atomic units {data structure, constraint type, motif}
     keep verifier specification (test generator + reference solution)

2. Recombination:
   sample target difficulty D
   pick a subset of atomic units that, combined, produce a problem of difficulty ≈ D
   instantiate concrete problem statement (templated NL + code skeleton)
   auto-derive verifier: stitch test generators from the constituent units
   verify a reference solution against the generated tests (sanity check)

3. Curriculum:
   ramp D over training to track the policy's improving capability
   maintain a fixed proportion of "frontier-difficulty" prompts
```

The verifier doesn't change — it's the same RLVR verifier (test pass / answer match) the policy already trains against. ADR's contribution is the *prompt distribution*.

Two design choices that matter:

- **Atomic granularity.** Too fine (single AST nodes) → combinatorial explosion of non-meaningful recombinations. Too coarse (whole problem skeletons) → no novelty. Mid-level units (e.g. "two-pointer constraint", "monotonic stack motif") work.
- **Difficulty calibration.** Estimated either by reference-solution length, expected number of motifs, or by running a small probe model and reading off its solve rate.

## Why it matters

- **Removes the prompt-supply ceiling.** Every recent reasoner hits the same wall: verifiers are cheap, but hard verifiable prompts are scarce. ADR makes prompt supply a *synthesis* problem rather than a *collection* problem.
- **Curriculum at the edge.** By tuning D online, the generated prompts stay at the policy's frontier of competence — exactly the regime RLVR needs for sustained gains.
- **Cross-domain.** Demonstrated on algorithmic programming, tool use, and data-science tasks; the framework is domain-agnostic given an atomic-unit grammar.

## Gotchas & tricks

- **The atomic grammar is the ML work.** Defining the units and their composability is more like data engineering than research, and quality of the grammar bounds quality of the output.
- **Avoid trivial recombinations.** Naive sampling produces many "two atoms welded together" problems whose solution is just both solutions concatenated. Penalize recombinations whose reference solution is structurally a concatenation.
- **Verifier brittleness.** Auto-derived test generators can be incomplete (they verify "a solution" but not "any solution"). Pair with a smaller, model-graded check for harder problems.
- **Contamination check.** Synthesized prompts that accidentally match public benchmarks (LiveCodeBench, etc.) re-introduce contamination. Run [decontamination](decontamination.md) on the synthesized set against benchmark suites.
- **Doesn't replace seed diversity.** ADR's output diversity is bounded by the variety of atomic units in the seed corpus — broaden the seeds, not just the recombinations.

## Sources

- Paper: *Combinatorial Synthesis: Scaling Code RLVR via Atomic Decomposition and Recombination* — Zheng et al., Inst. of Software CAS, 2026 — [arXiv:2605.31058](https://arxiv.org/abs/2605.31058) — introduces ADR, demonstrates curriculum scaling past heuristic seed expansion.
