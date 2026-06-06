# RL Task Synthesis
*Depth — generating novel verifiable tasks for RLVR by decomposing existing tasks into atomic primitives and combinatorially recombining them.*

**TL;DR:** RLVR's bottleneck is no longer the algorithm — it's the supply of *hard, verifiable* prompts near the model's competence edge. Heuristic seed expansion (paraphrase, mutate constants) runs out of novelty quickly. **Atomic Decomposition and Recombination (ADR, 2026)** decomposes existing tasks into atomic elements (functions, constraints, test specs) and recombines them under controlled rules to produce verifiable tasks that are genuinely novel and harder than the seeds. Demonstrated for code RLVR; the recipe generalizes wherever atoms are well-typed.

**Prereqs:** [rlvr](rlvr.md), [rl-prompt-curation](rl-prompt-curation.md)
**Related:** [_rl](_rl.md), [grpo](grpo.md)

---

## What it is

RLVR trains on `(prompt, verifier)` pairs. The training distribution determines what the model learns. Two ways to grow it:

- **Curation** — pick the best prompts from existing ones (importance sampling, difficulty filters). Counterpart: [rl-prompt-curation](rl-prompt-curation.md).
- **Synthesis** — generate new prompts that don't exist in the seed set. ADR is the synthesis side.

Heuristic synthesis (paraphrase, vary constants, change problem statement) preserves the structure of the seeds and therefore caps difficulty and diversity. ADR breaks this cap by working *below* the task level.

## How it works

Two stages:

**1. Atomic Decomposition.** Parse existing verified tasks into reusable atoms:

- *Function atoms* — small composable operations (sort, filter, deduplicate, aggregate).
- *Constraint atoms* — input/output shape, complexity bound, edge-case requirement.
- *Test atoms* — assertion templates that verify a piece of behavior.

The decomposition is task-class specific. For code: AST-level extraction. For math: lemma-level decomposition.

**2. Controlled Recombination.** Combine atoms under rules that *preserve verifiability*:

- Composition rules ensure the recombined task can still be checked by a programmatic verifier.
- Difficulty rules tune how many atoms / which interaction patterns are allowed.
- Diversity rules reject combinations near existing tasks (embedding-based dedup).

Output: a new `(prompt, verifier)` pair where the verifier is the composed assertion set from the atoms. Verifiability is by construction — no LLM-judge step needed.

## Why it matters

- **Difficulty scales with combination count.** Recombining 5 atoms produces tasks meaningfully harder than the 5 seed tasks. Difficulty knob built into the synthesis process.
- **Novelty beyond paraphrase.** Atomic recombination produces tasks structurally absent from the seed set, which paraphrase-based synthesis can't.
- **Downstream RLVR gains across domains.** ADR-trained models improve on algorithmic programming, tool use, and data-science tasks — the synthesis transfers beyond the immediate code-task family.

## Gotchas & tricks

- **Atom quality dominates.** Bad atoms → bad recombinations. The decomposition step needs careful schema design, often per task class.
- **Verifier-side correctness is brittle.** A composed verifier might accept an unintended solution if a missing atom-level constraint isn't enforced. Treat the verifier as a software engineering artifact, not a heuristic.
- **Combinatorial explosion.** Without diversity / novelty filters, the synthesis space is enormous and mostly garbage. Filter aggressively before passing to RLVR.
- **Doesn't replace curation.** Even with ADR, you still want a curriculum / difficulty ramp; treat synthesis as the prompt-supply step and curation as the scheduling step.
- **Plausibly applicable to math.** Math lemmas and theorem-prover tactics are natural atoms; the framework should port with adapter work.

## Sources

- Paper: *Combinatorial Synthesis: Scaling Code RLVR via Atomic Decomposition and Recombination* — Zheng et al., 2026 — [arXiv:2605.31058](https://arxiv.org/abs/2605.31058) — primary source.
- Paper: *Tülu 3* — AI2, 2024 — earlier prompt-curation-heavy RLVR pipeline that ADR's synthesis complements.
