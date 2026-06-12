# Long-Horizon SWE Data (DeNovoSWE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A self-supervised pipeline that generates **whole-repository construction trajectories** from documentation-style specs, then filters them via difficulty-aware selection. The pipeline uses **divide-and-conquer** (a planner breaks specs into subtasks, executors implement, a critic loops in repair) inside a sandbox that verifies correctness automatically — no human labels. The resulting **DeNovoSWE** dataset (4,818 instances) used to SFT Qwen3-30B-A3B raises the BeyondSWE-Doc2Repo score from **5.8% → 47.2%**. Generalizes to any sandbox-verifiable agentic task.

**Prereqs:** [../data/_data-curation.md](../data/_data-curation.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) · [README.md](README.md)

---

## What it is

Code agents have been moving from "fix this bug in this file" (SWE-bench) to "build this entire repo from a documentation spec" (BeyondSWE-Doc2Repo). Training data for the harder problem barely exists — whole-repo construction trajectories are long, multi-step, and hard to verify, so human-labeled data caps out at small sizes.

DeNovoSWE solves the data shortage by letting agents *generate the trajectories themselves* under a sandboxed verifier. Three structural moves:

1. **Divide-and-conquer planning.** Large planner agents break documentation specs into module-level subtasks.
2. **Critic-repair loops.** Executor agents implement; a critic agent reviews; the executor revises until tests pass.
3. **Difficulty-aware filtering.** Trajectories are scored on difficulty (something like test-pass complexity × repo size); the filter rebalances the dataset away from over-represented easy examples.

---

## How it works

### The synthesis loop

For each documentation spec:

```
1. Planner agent decomposes spec → module-level task list.
2. For each task, an executor agent writes code in a sandboxed worktree.
3. A critic agent runs tests and reads outputs.
4. If tests fail, the critic produces a repair prompt; back to step 2.
5. Once all tasks pass, the full trajectory (plan + edits + repairs) is saved.
```

The sandbox is the verifier: only trajectories that *actually produce a working repository* are kept. False positives (claimed success but broken code) are filtered by re-running tests after the fact.

### Why divide-and-conquer

Whole-repo construction is too long-horizon for a single executor to keep coherent context over. Divide-and-conquer keeps each executor's task small and verifiable in isolation; the planner handles cross-task coherence.

### Difficulty-aware filtering

Naively kept trajectories skew easy: simple repos succeed quickly, complex ones often fail. The filter estimates per-trajectory difficulty (combination of test count, failure-rate during synthesis, and final-repo complexity) and resamples so the dataset has balanced difficulty coverage. Prevents the model from over-fitting to trivial whole-repo tasks.

### Result: 4,818 instances

Each instance is a (documentation spec, working repository) pair, with the full intermediate trajectory available for SFT or RL.

---

## Why it matters

- **5.8% → 47.2% on BeyondSWE-Doc2Repo** after Qwen3-30B-A3B SFT on DeNovoSWE. ~8× absolute gain, SFT-only, no RL stage.
- **Verifies data scarcity was the binding constraint** — not architecture, not RL stage, just lack of training data on long-horizon construction.
- **Recipe generalizes.** Divide-and-conquer + critic-repair + difficulty-aware filtering works for any agentic task with a sandbox-verifiable success criterion: full-stack web apps, ML pipeline construction, infra-as-code, etc.
- **No human labels needed.** Pipeline is self-supervised against the sandbox; scales with compute.

---

## Gotchas & tricks

- **Critic bias.** The critic is itself an LLM agent; if it shares failure modes with the executor (same base model), it'll miss the same bugs. Use a different model class for the critic when budget allows.
- **Sandbox coverage matters.** The verifier is only as good as the tests included in the sandbox. If specs come with weak tests, the synthesized trajectories will succeed but the resulting repos may be subtly broken — and you'll have trained on subtly-broken examples.
- **Difficulty estimation is noisy.** Test count is a poor proxy on its own; repos with many trivial tests look "hard" but aren't. Combine multiple difficulty signals.
- **Divide-and-conquer assumes decomposability.** Specs that require cross-cutting reasoning (e.g. a global architectural choice that affects every module) break the planner's clean decomposition. The paper reports lower yield on these cases.
- **Trajectory length is the cost driver.** Each trajectory may involve dozens of edits and tens of test runs. Synthesis cost per instance is large; budget accordingly.

---

## Sources

- Paper: *DeNovoSWE: Scaling Long-Horizon Environments for Generating Entire Repositories from Scratch* — Zhang, Chen, Meng, Zhou, Song, Wen, Jia (Renmin U / ByteDance), 2026 — [arXiv 2606.10728](https://arxiv.org/abs/2606.10728).
- Concept: SWE-bench, BeyondSWE-Doc2Repo — long-horizon SWE evaluation benchmarks.
