# Agentic Synthetic Data
*Depth — treat synthetic-data generation as an iterative agentic loop with explicit quality measurement and recipe updates.*

**TL;DR:** Classical synthetic-data pipelines (Self-Instruct, Evol-Instruct, distillation-from-teacher) use a fixed prompt template and a one-shot generator. *Agentic synthetic data* replaces that with an LLM agent that generates → evaluates quality → analyzes failures → rewrites its own generation recipe in a feedback loop. Autodata (Meta FAIR, 2026) is the canonical recipe: an orchestrator coordinates a Challenger (writes problems), a Weak Solver, a Strong Solver, and a Verifier/Judge — the weak/strong-solver gap is the explicit learning signal, and the orchestrator's own prompt can be meta-optimized.

**Prereqs:** [_data-curation](_data-curation.md), [quality-filtering](quality-filtering.md)
**Related:** [decontamination](decontamination.md), [post-training/rejection-sampling](../post-training/rejection-sampling.md), [post-training/rl-prompt-curation](../post-training/rl-prompt-curation.md)

---

## What it is

A multi-agent pipeline whose output is a training corpus (instructions, QA pairs, problem-solution pairs). At least three roles:

- **Challenger.** Writes a candidate example given source material and a recipe.
- **Solver(s).** Attempt to solve it. A "weak" model is expected to struggle; a "strong" model is expected to succeed. The gap between them is the discriminative signal.
- **Verifier / Judge.** Scores quality and writes feedback that the orchestrator can act on.

Around them, an **orchestrator** runs the loop: generate a batch → measure weak/strong gap and verifier scores → analyze the failing examples → update the recipe (the prompt that the Challenger uses) → repeat.

## How it works

The Autodata implementation ("Agentic Self-Instruct") uses these stages:

1. **Generate.** Challenger produces a batch of (prompt, expected solution) pairs grounded in a source document.
2. **Solve.** Run both Weak and Strong solvers on each example. Record their answers.
3. **Score.** Verifier judges correctness and quality.
4. **Filter.** Keep examples where (Strong succeeds) ∧ (Weak fails) — i.e. discriminative under the current solver gap. Drop trivial or impossible examples.
5. **Analyze.** Aggregate the verifier's feedback; cluster failure modes; update the Challenger's recipe with explicit rules ("require numeric answers in <answer></answer>", "avoid leaking source filenames").
6. **Iterate.** Repeat until quality plateaus.

The orchestrator's own prompt can also be optimized via prompt evolution (Autodata reports validation pass rate moving from 62.1% → 79.6% across 126 iterations).

## Why it matters

- **Closes the outer loop.** Self-Instruct and Evol-Instruct fix the generation recipe; Autodata-style pipelines let the recipe adapt to the actual quality signal.
- **Inference-time compute → better training data.** Spending more inference compute on data curation translates to better post-training. Quantified: a 4B model trained on Autodata-generated legal QA outperforms a 397B baseline on PRBench-Legal.
- **Works across verifiable and non-verifiable tasks.** Autodata reports gains on math (verifiable), CS-paper QA (semi-verifiable), and legal reasoning (non-verifiable). The weak/strong gap is the unifying signal when no rule-based verifier is available.
- **Generalizes the SFT/RLHF data pipeline.** Composes with RLVR (use Autodata to generate the prompt distribution that RLVR then optimizes against).

## Gotchas & tricks

- **Weak/strong gap dynamics.** If the weak model is *too* weak, every example is "discriminative" but most aren't useful. Pick a weak model that's mid-tier.
- **Verifier reward hacking.** A learned verifier can be gamed by the Challenger over enough iterations. Periodically swap in a held-out verifier to detect drift.
- **Source-grounded vs free-form.** Source-grounded generation (Autodata's setting — papers, legal docs) is more controllable. Free-form generation needs strong filtering against contamination.
- **Compute-aware budgets.** The loop is expensive (multi-agent rollouts per data point). Budget by *training tokens produced*, not by iteration count.
- **Decontamination on the way out.** Run standard decontamination (n-gram match against eval sets) on the final corpus. Agents will happily regenerate eval-set problems.
- **Meta-optimization helps but plateaus.** The orchestrator-prompt evolution gives a one-time bump; don't expect compounding gains.

## Sources

- Paper: *Autodata: An agentic data scientist to create high quality synthetic data* — Whitehouse, Wu, Nie, Saha, et al. (Meta FAIR), 2026 — [arXiv 2606.25996](https://arxiv.org/abs/2606.25996).
- Predecessor: *Self-Instruct* — Wang et al., 2022 — [arXiv 2212.10560](https://arxiv.org/abs/2212.10560).
- Predecessor: *Evol-Instruct (WizardLM)* — Xu et al., 2023 — [arXiv 2304.12244](https://arxiv.org/abs/2304.12244).
