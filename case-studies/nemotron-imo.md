# Case Study: Nemotron IMO-Gold Recipe

*NVIDIA's open post-training + test-time-compute pipeline that took Nemotron 3 Ultra to gold at IMO 2026 (30/42 points) with no formal prover, no external tools, and no internet — using only natural-language proofs and released, reproducible checkpoints.*

**Related concepts:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md) · [orm](../post-training/reasoning/orm.md) · [prm](../post-training/reasoning/prm.md) · [rejection-sampling](../post-training/rejection-sampling.md) · [rlvr](../post-training/rlvr.md) · [generate-verify-refine](../post-training/reasoning/generate-verify-refine.md) · [nemotron-imo-bench](../evaluation/nemotron-imo-bench.md) · [long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What this is

The Nemotron IMO paper (arXiv 2609.10712) is an *open recipe*, not a new base model release. It takes NVIDIA's Nemotron 3 Ultra as the frozen backbone and shows that:

1. Two **RL-post-trained specialists** on top of the general-availability checkpoint
2. Plus an **iterative generate-verify-refine loop** running entirely in natural language
3. Plus a **separate high-compute selection stage** for the final submission per problem

is enough to score **30/42 at IMO 2026 — gold-medal threshold** — with no formal prover, no external tools, no internet, and no per-problem human oversight.

The paper is worth reading as a full-system case study for two reasons:
- It's the first *open* reproduction of IMO-gold-caliber math reasoning with a documented post-training and inference pipeline.
- It cleanly separates what comes from post-training vs. what comes from test-time compute — a template other frontier reasoning attempts can follow.

---

## Architecture at a glance

```
                Nemotron 3 Ultra (GA base)
                          │
                          ├────────────────────────────┐
                          ▼                            ▼
              SFT + RL specialist A          SFT + RL specialist B
              (Solver — proposes proofs)     (Verifier / Refiner — critiques & repairs)
                          │                            │
                          └──────────────┬─────────────┘
                                         ▼
                          ┌──────────────────────────────┐
                          │ Iterative generate-verify-   │
                          │ refine loop (natural language)│
                          │  - K candidate proofs        │
                          │  - verify each               │
                          │  - refine or discard         │
                          │  - repeat R rounds           │
                          └──────────────┬────────────────┘
                                         ▼
                          High-compute selection stage
                          (per-problem final choice)
                                         ▼
                                 Submitted proof
```

Three checkpoints ship: the GA Nemotron 3 Ultra, plus specialists A and B. All three are RL-post-trained on math distributions but with different targets — specialist A optimizes end-to-end proof quality; specialist B optimizes step-level critique and repair.

---

## Training recipe

### Stage 1 — Supervised fine-tuning

- **Base:** Nemotron 3 Ultra (general-availability weights). No pretraining changes.
- **Data:** curated olympiad-style proofs + step-by-step natural-language reasoning traces. Data mix + generation code released.
- **Two specialist targets:** the same SFT corpus is used to seed both specialists, but with different task templates — proof-generation prompts for A, critique/repair prompts for B.

### Stage 2 — RL specialization

- **Reward:** correctness-of-final-answer for A (with heavy weight on proofs that pass the verifier's checks). For B, a per-step critique reward that measures whether B's critique identifies the actual flawed step in a corrupted candidate.
- **Loop:** standard RL post-training over sampled proofs; specialists co-evolve with the paper's own verifier signals.
- **Long-CoT preserved.** No length penalty during specialization — proofs are allowed to run long, which the results depend on.

Both specialist checkpoints are released alongside the paper.

---

## Test-time-compute pipeline

Two nested loops. The **inner loop** is the generate-verify-refine cycle (see [generate-verify-refine.md](../post-training/reasoning/generate-verify-refine.md)):

1. Specialist A proposes K candidate proofs for the problem.
2. Specialist B verifies each: passes or names the flawed step.
3. For each verified-flawed candidate, specialist B (or A conditioned on B's critique) produces a refined variant.
4. Re-verify. Keep refined candidates that now pass. Repeat for R rounds.

The **outer loop** is the high-compute per-problem selection stage: for each IMO problem, run the inner loop with a large K and R, gather the surviving candidates, and use a separate high-compute selection pass to pick the final submission (details in the paper — a mix of verifier scoring, self-consistency, and cross-candidate comparison).

Everything happens in natural language. No solver. No tool calls. No internet.

---

## Results

| | Score / Metric |
| --- | --- |
| IMO 2026 total (out of 42) | **30** (gold threshold) |
| Problems fully solved | 4 of 6 |
| Setting | Natural-language proofs; no prover, no tools, no internet |
| Selection compute | High per-problem inference budget |

Plus the paper releases:
- Both specialist checkpoints (SFT+RL) on top of Nemotron 3 Ultra
- Training data + generation code
- Inference-time search + selection code
- The submitted solutions per problem
- **Nemotron-IMO-Bench** — 200 novel olympiad-level problems (see [nemotron-imo-bench.md](../evaluation/nemotron-imo-bench.md))

---

## Key takeaways

1. **Specialize, don't monolithize.** Two specialists (proposer + verifier/refiner) are stronger than one generalist for reasoning search. The verifier gains come from RL on the critique task, not from more solver training.

2. **Natural-language proofs are competitive with formal-prover setups** at IMO-gold caliber. If you have a strong verifier specialist, you don't need Lean/Coq for a passing performance.

3. **The generate-verify-refine loop is the load-bearing pattern.** Naive best-of-N sampling from specialist A alone underperforms; the refine step (rewrite around the identified flawed step) is where much of the lift comes from.

4. **Selection is a separate compute budget.** Splitting inner-loop generation from outer-loop selection lets you spend the compute where it matters — most problems are settled by the inner loop, hard problems consume most of the outer loop.

5. **RL post-training on a strong GA base beats retraining.** No architecture change, no new pretraining — just two targeted SFT+RL passes on top of a released checkpoint. The specialists are cheap to produce relative to the base's cost.

6. **Openness makes it a recipe.** Data, code, checkpoints, and a released benchmark are what turn a headline (IMO-gold score) into a *reproducible* pattern others can adapt.

---

*Pairs well with:* [DeepSeek-R1](deepseek-r1.md) for the long-CoT RL foundation, and [Kimi K1.5](kimi-k1-5.md) for another RL-driven reasoning system.
