# AdvancedMathBench (ProverBench + VerifierBench)
*Depth — proof generation and proof verification at undergraduate and doctoral qualifying-exam level.*

**TL;DR:** A benchmark suite that pushes math evaluation past MATH500/AIME's high-school-and-olympiad ceiling. **ProverBench** contains 296 advanced-math proof-generation problems (undergraduate + doctoral qualifying-exam splits); **VerifierBench** contains 888 model-generated proof trajectories paired with expert ground truth for evaluating whether LLMs can *judge* proofs. Ships with a trained automatic verifier that gives both correctness verdicts and fine-grained per-step error assessments.

**Prereqs:** [math500.md](math500.md), [aime.md](aime.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md), [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md)

---

## What it is

Existing math benchmarks (MATH500, GSM8K, AIME) evaluate on **final-answer correctness** at competition or high-school level. Two gaps:

- **Scope** — advanced math (real analysis, abstract algebra, topology at UGD level; qualifying-exam problems at QE level) is under-covered.
- **Evaluation granularity** — final-answer scoring misses the reasoning; a right answer via a wrong proof passes, a right proof with a slip fails.

AdvancedMathBench addresses both:

- **ProverBench (296 problems).** Advanced-math proof-generation problems, split into **UGD** (undergraduate) and **QE** (doctoral qualifying-exam).
- **VerifierBench (888 trajectories).** Model-generated proof attempts, each labeled by experts as valid or invalid, with error typing where invalid. Directly tests LLMs as proof judges.
- **Trained automatic verifier.** A verifier model trained on large-scale expert annotations of proof trajectories, producing both a binary correctness verdict and fine-grained assessments of proof errors. Strong agreement with human experts on held-out sets.

## How it works

### ProverBench evaluation

For each problem, a candidate model produces a proof. The trained verifier scores the proof against a rubric:

- Overall correctness (0/1).
- Per-step error classes (missing case, unjustified step, wrong lemma cited, symbolic slip).

Aggregate metric: accuracy on UGD and QE splits separately.

### VerifierBench evaluation

Given a (problem, model-generated proof) pair with a hidden expert label, the model under test must produce a verdict (valid / invalid) and, ideally, a supporting rationale. Metric: **Balanced F1** over the label distribution, plus true-negative rate (does the model catch *wrong* proofs?).

### The verifier itself

The trained verifier is a first-class artifact: it's what lets ProverBench be scaled beyond small human-graded sets. Its agreement with human experts on held-out proof trajectories is the paper's own external validity check.

## Why it matters

- **Fills the top-end reasoning-eval gap.** After MATH500 and AIME saturate for frontier models, there's a benchmark that still moves.
- **Directly stress-tests LLM-as-judge for reasoning.** Every RLVR pipeline, self-critique loop, and process-reward model implicitly assumes an LLM can judge its own or others' work. VerifierBench measures that assumption at proof-level fidelity.
- **Frontier models are still weak.** Best model scores **75.8 / 66.1** on UGD/QE proof generation and **65.1 Balanced-F1** on verification, with low true-negative rates. There is measurable headroom.

## Gotchas & tricks

- **Automatic-verifier ceiling.** Reported ProverBench scores are only as reliable as the trained verifier. When comparing models with tiny gaps, cross-check on the human-labeled subset.
- **Contamination risk.** Doctoral qualifying-exam problems have well-known solutions on the open web. Watch for train-time exposure when using models with unclear cutoffs; run the contamination check.
- **Not a good RLVR target — yet.** ProverBench rewards long, structured proofs, not short verifiable answers. Using it directly as an RLVR reward would require running the trained verifier at rollout speed; not designed for that.
- **QE split is small relative to variance.** With only ~100+ problems per split, single-run scores are noisy; report averages over ≥3 seeds.

## Sources

- Paper: *AdvancedMathBench: A Benchmark Suite for Advanced Mathematical Proof Generation and Verification* — Kong et al., Shanghai AI Lab (Intern Large Models), 2026 — arXiv:2607.11849.
- Related benchmarks: MATH500 (Hendrycks et al.) — see [math500.md](math500.md); AIME (competition-level short-answer) — see [aime.md](aime.md).
- Related: process reward models — see [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md).
