# Agent Benchmark Synthesis (TASTE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Agent benchmarks built scenario-first (write a natural-language task, then map it to tool sequences) capture only a narrow slice of the tool-use space and saturate quickly. **TASTE reverses the pipeline**: sample valid tool sequences first using an Adaptive Contrastive n-gram model trained on LLM-judged validity signals, cluster them for coverage, then instantiate each cluster into a benchmark task and **iteratively evolve difficulty**. Result: τ^c-Bench — a τ^2-Bench extension where Gemini-3-Flash drops from 0.82–0.94 to 0.28–0.61 across three domains, with 2× the unique tool combinations.

**Prereqs:** [README](README.md), [../agents/README.md](../agents/README.md)
**Related:** [../evaluation/livecodebench.md](livecodebench.md)

---

## What it is

Tool-using agent benchmarks (τ-Bench, τ²-Bench, etc.) are saturating: frontier models score 80–95% on tasks designed two years ago. The standard remedy — write harder scenarios — is expensive and biased: humans naturally think in *intents* ("book me a flight, then check the weather") which map to a narrow part of the tool-use combinatorial space. TASTE inverts the dependency: enumerate valid *tool sequences* directly via a learned generator, then write the natural-language framing afterwards. This systematically covers tool-use patterns no human would have prompted for.

---

## How it works

### Pipeline

```
1. Validity model: Adaptive Contrastive n-gram model trained on LLM-judged
   (sequence_prefix → next_tool) preference signals. Captures which tool
   transitions are syntactically and semantically plausible in the target
   environment.

2. Generation: sample diverse tool sequences from the validity model.
   Sampling temperature controls trade-off between coverage and validity.

3. Clustering: cluster the sampled valid sequences (e.g. by edit distance
   or tool-multiset embedding). Pick a representative per cluster for
   coverage breadth.

4. Instantiation: turn each representative sequence into a full benchmark
   task — natural-language user intent + initial state + expected outcome.
   This is the only step that requires per-task LLM annotation.

5. Difficulty evolution: iteratively rewrite tasks (add distractors,
   constraints, alternative valid paths) until candidate agents fail at
   a target rate. Mirrors evolutionary instruction-tuning recipes.
```

### Why "Adaptive Contrastive n-gram"

A pure n-gram model would over-fit observed sequences; a transformer-based generator would be expensive to train per environment. The contrastive n-gram trains on **(positive, negative)** transition pairs scored by an LLM judge ("is this a plausible next tool given this prefix?"). "Adaptive" = the n-gram context window is learned per-environment based on dependency length.

---

## Why it matters

- **Diagnoses why benchmarks saturate.** It isn't just that models improved — the benchmark *construction process* hits a coverage ceiling. TASTE's 2× tool-combination coverage at the same per-task cost shows the construction is, not the model, the bottleneck.
- **Concrete reusable recipe.** Reverse construction (sequence-first → cluster → instantiate → evolve) ports to other agentic benchmarks: web browsing, coding agents, computer use.
- **Decouples coverage from human authoring.** Human authors can refine and instantiate; they don't need to imagine the combinatorial space.
- **Difficulty evolution is the right level of abstraction.** Rather than picking a single difficulty per task, evolve tasks toward a target failure rate against current frontier models. Self-correcting as models improve.

---

## Gotchas & tricks

- **Quality of the validity model gates everything.** A weak validity model produces nonsense sequences; downstream instantiation can't fix that.
- **LLM-judged validity signals are noisy.** The paper uses self-consistency / multiple judges to denoise; budget LLM-judge cost into the per-task synthesis budget.
- **Difficulty evolution can overfit to a single judge model.** If you evolve to fail Gemini-3-Flash, you may not transfer-fail Claude-Opus-4 or GPT-5.5. Evolve against a panel.
- **Trickier with stateful environments.** TASTE was demonstrated on τ²-Bench's three domains (retail, airline, telecom) which have well-defined APIs. For browser-use environments, sampling valid sequences requires environment rollout — much more expensive.
- **Risk: contamination.** If the sequence generator was trained on data that includes evaluation models' training set, expect some leakage. Mitigate via held-out tool sequences during validity training.
- **Per-task instantiation is the cost driver.** The cheap part is sampling sequences (millions per dollar); the expensive part is the LLM-rewrite that turns a sequence into a natural-language task.

---

## Sources

- Paper: *A Matter of TASTE: Improving Coverage and Difficulty of Agent Benchmarks* — Keren, Calderon, Yehudai, Perlitz, Shmueli-Scheuer, Reichert — Technion, 2026 — [arXiv:2605.28556](https://arxiv.org/abs/2605.28556).
- Background: *τ-Bench* and *τ²-Bench* — Sierra / collaborators, 2024 — the benchmarks TASTE extends.
- Adjacent: evolutionary instruction-tuning (WizardLM, Evol-Instruct, 2023) — same difficulty-evolution pattern applied to instruction data instead of agent tasks.
