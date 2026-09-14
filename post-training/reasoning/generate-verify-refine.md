# Generate-Verify-Refine
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An iterative inference-time search pattern for reasoning: **generate** candidate solutions with a solver specialist, **verify** each with a critique specialist that either passes it or names the flawed step, **refine** flagged candidates conditioned on the critique, then re-verify. Both models are RL-post-trained natural-language LLMs — no formal prover, no tools. This is the load-bearing loop in the Nemotron IMO-gold recipe.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [../_rl](../_rl.md)
**Related:** [orm](orm.md), [prm](prm.md), [../rejection-sampling](../rejection-sampling.md)

---

## What it is

Best-of-N sampling on a strong solver saturates: many candidates all fail in the same way and pass a light verifier, so the score plateaus with more samples. Generate-verify-refine adds a **repair step** — the verifier doesn't just accept/reject; it names *which step is wrong*, and a refiner rewrites the candidate around that step. Wrong candidates become chances to explore neighboring reasoning traces, not deadweight.

The pattern applies whenever you have (a) a solver that can produce diverse candidates and (b) a verifier that can localize errors, both natural-language and both trained rather than symbolic.

## How it works

- **Two specialists.** A *solver* (RL-post-trained for end-to-end proof quality) and a *verifier/refiner* (RL-post-trained on the critique-and-repair task). Both share a base model; they diverge only in post-training.
- **Inner loop.**
  1. Solver samples K candidate proofs for the problem.
  2. Verifier scores each; for failing candidates, emits a natural-language critique naming the flawed step.
  3. Refiner (verifier acting as refiner, or solver conditioned on the critique) produces a refined variant of each flagged candidate.
  4. Re-verify. Accept newly-passing candidates.
  5. Repeat for R rounds.
- **Selection stage (outer loop).** After R rounds, the surviving candidate pool is passed to a separate higher-compute selection procedure (verifier scoring + self-consistency + cross-candidate comparison) that picks the final answer.
- **Everything in natural language.** No formal proof assistant. No tool execution. The verifier's supervision came from RL on the critique task, not from an external oracle.

## Why it matters

The pattern is what turns a strong solver + strong verifier into a *reasoning system* rather than two separate models. In the Nemotron IMO recipe it delivers gold-medal (30/42) at IMO 2026 with no tools. It also cleanly separates two compute budgets — the inner loop (per-candidate refinement) from the outer loop (per-problem selection) — so you can scale each independently.

## Gotchas & tricks

- **Refine step must be flaw-conditional.** Blind re-sampling wastes compute; the critique must be threaded through so the refiner rewrites around the identified step.
- **Verifier quality is the bottleneck.** Weak verifiers accept nonsense — most gains come from RL-post-training the verifier on hard critique tasks, not from scaling the solver.
- **Diversity in the initial K matters.** If solver rollouts collapse to one trace, the loop degenerates. Use high temperature or diverse prompt templates for the seed pool.
- **Refined candidates need re-verification.** Accepting on the refiner's word (self-verification) leaks confidence bias; always re-run the verifier.
- **Rounds R hits diminishing returns fast.** Most gains land in the first 1–2 refinement rounds; beyond 3–4, spend compute on outer-loop selection instead.
- **Selection ≠ verification.** The final selection stage combines verifier score with self-consistency and pairwise comparison — a single verifier pass under-selects on IMO-hard problems.

## Sources

- Paper: *An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics* — Moshkov, Ge, Armstrong, Du, Mahdavi, Gitman (NVIDIA), 2026 — https://arxiv.org/abs/2609.10712
- Related: process-reward and outcome-reward verifier literature ([prm](prm.md), [orm](orm.md)).
