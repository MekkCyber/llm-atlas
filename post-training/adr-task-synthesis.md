# ADR — Atomic Decomposition and Recombination for Code RLVR
*Depth — synthesise hard, verifiable code tasks by decomposing existing tasks into atoms and recombining them under control.*

**TL;DR:** Code RLVR is gated by the supply of *hard, verifiable* tasks. Heuristic seed-expansion ("rewrite this problem with a different cover story") quickly saturates in difficulty and diversity. **ADR** (Zheng et al., 2026) decomposes seed tasks into reusable atomic elements — algorithmic primitives, constraints, inputs, output structures — and recombines them with explicit controls, generating novel problems that hit the model's competence edge without breaking verifiability. The synthesised data outperforms prior baselines on originality, difficulty, diversity, and test quality, with consistent downstream gains across algorithmic programming, tool use, and data science.

**Prereqs:** [post-training/rlvr.md](./rlvr.md), [post-training/rl-prompt-curation.md](./rl-prompt-curation.md)
**Related:** [post-training/grpo.md](./grpo.md), [evaluation/livecodebench.md](../evaluation/livecodebench.md), [evaluation/codeforces-benchmark.md](../evaluation/codeforces-benchmark.md)

---

## What it is

Atomic decomposition splits each seed task into orthogonal axes — what's the algorithm, what's the constraint set, what's the input distribution, what's the expected output shape, what verifies a solution. Recombination samples atoms across these axes to assemble a new task whose verification scaffold is inherited from its parts.

The point is to break the implicit covariance heuristic expansion preserves: if you only paraphrase a problem, you get a near-duplicate; if you cross algorithms with novel constraints with novel inputs, you get a genuinely new problem whose difficulty is controllable.

## How it works

1. **Atomise seeds.** A library of verified code problems is annotated by axis: `algorithm`, `data_structure`, `constraint`, `input_distribution`, `output_format`, `verifier_kind`. Annotation is mostly LLM-driven with rule-based sanity checks.
2. **Recombine under control.** A sampler picks atoms across axes subject to compatibility constraints (a `graph` algorithm requires a `graph`-shaped input). Controls let the synthesiser target a difficulty band or a downstream skill.
3. **Synthesise problem + tests.** An LLM emits the problem statement, a reference solution, and a test suite. Atoms carry their verifiers, so the test scaffold is mostly assembled, not freshly generated.
4. **Filter for verifiability.** Generated problems whose tests are flaky or whose reference solution fails are discarded. The bar is *deterministically verifiable*, not just plausibly correct.
5. **Train RLVR on the synthesised mix.** Use the standard RLVR loop (rule-based reward + GRPO).

## Why it matters

- **Lifts the data ceiling for code RLVR.** Without a way to scale *difficulty*, RLVR plateaus the moment the model can solve the seed pool. ADR moves the ceiling explicitly.
- **Controllable difficulty.** Because difficulty is composed from atoms with known costs, you can target the band where the current policy is at its competence edge — the sweet spot for RL.
- **Transfers across downstream domains.** Tool-use and data-science gains, not just competitive programming. The atom library is broad enough that the synthesised distribution covers more than its seed.
- **Higher test quality.** Inherited verifiers are far less likely to be wrong than freshly LLM-authored ones, which has been the dominant failure mode of LLM-synthesised code RL data.

## Gotchas & tricks

- **Atom annotation quality is the cap.** If atoms are mis-tagged, recombination produces incoherent problems; allocate budget to verifying the atom library before scaling.
- **Compatibility constraints are non-trivial.** Naively crossing all atoms produces 90%+ invalid problems. Hand-curated compatibility rules are the difference between 10% and 60% yield.
- **Verifier flakiness compounds.** A single non-deterministic test (clock, network, randomness) leaks into RLVR reward and the policy learns to game it. Static-analysis flake detectors before training.
- **Difficulty drift.** The synthesiser, conditioned on a strong LLM, biases toward problems the LLM finds easy. Periodically re-rank against the current policy and shift the target band.
- **Don't confuse with self-instruct.** Self-instruct paraphrases; ADR recombines structure. The two stack — paraphrase the surface form *after* recombining the atoms.

## Sources

- Paper: *Combinatorial Synthesis: Scaling Code RLVR via Atomic Decomposition and Recombination* — Zheng et al. (Institute of Software, Chinese Academy of Sciences), 2026 — [arXiv:2605.31058](https://arxiv.org/abs/2605.31058).
