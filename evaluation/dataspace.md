# DataSpace

*Depth — benchmark for data agents that produce verifiable tabular answers over heterogeneous local workspaces (CSV, JSON, SQLite, Markdown, PDF, video).*

**TL;DR:** DataSpace (HKUST(GZ) / Tsinghua, 2026) is a benchmark for the "data analyst agent" pattern: natural-language question over an organizational workspace, tabular answer required, deterministic verification. 410 cross-language tasks, 7,439 artifacts across CSV / JSON / SQLite / Markdown / PDF / video, 15.01 GB total. Served as the official evaluation for the KDD Cup 2026 Data Agents track.

**Prereqs:** *(basic tool-using LLM agent familiarity)*
**Related:** [README.md](README.md), [../agents/README.md](../agents/README.md)

---

## What it is

A benchmark focused on three axes prior data-agent benchmarks left disconnected:

- **Heterogeneous evidence.** Answers require discovering relevant evidence scattered across many file types in one workspace, not one modality at a time.
- **Complete tabular output.** The agent must return a *complete* table matching the requested schema, not free text.
- **Deterministic grading.** Tabular outputs allow exact comparison to a ground-truth table — no LLM judge in the loop.

The workspace scope is task-local: each task ships its own workspace of artifacts; the agent has to figure out which are relevant.

---

## How it works

**Task.** Each item is `(question_in_natural_language, workspace_of_artifacts)`. The agent returns a table.

**Grading.** Exact schema and value match against ground truth. Deterministic — no rubric, no judge.

**Task languages.** Cross-language (multiple natural languages for the query), forcing agents to handle both language robustness and multi-format retrieval.

**Composition.** 410 tasks over 7,439 files totalling 15.01 GB, spanning CSV / JSON / SQLite / Markdown / PDF / video.

**Usage.** Served as the KDD Cup 2026 Data Agents for Complex Data Analysis competition — external validation that the benchmark is discriminative at frontier scale.

## Why it matters

- **Cuts the "does it sound right" trap.** Data-QA benchmarks with free-text answers are hard to grade cheaply and noisy. Requiring complete tabular output makes exact grading feasible for open-ended questions.
- **Reflects real deployment.** Enterprise data-agent asks are heterogeneous workspaces of mixed-quality artifacts; DataSpace is the first benchmark to sit inside that regime end-to-end.
- **Video + structured formats in one benchmark.** Multimodal agents no longer have to choose which modality to demonstrate on.

## Gotchas & tricks

- **Schema strictness.** Exact tabular match is unforgiving of column-order or type mismatches; agents that reason correctly but miss the schema look wrong. Practical systems add a schema-normalization post-step.
- **Workspace size.** 15 GB of artifacts across 410 tasks means individual workspaces can be large — retrieval strategy dominates performance for many models.
- **Cross-language noise.** Some tasks are in languages the agent may weakly understand; separate reporting by task language distinguishes language-competence from data-competence failures.
- **Benchmark → training data leakage.** Public benchmark data can leak into future training runs; watch the reported dates and use held-out splits for regression testing.

## Sources

- Paper: *DataSpace: Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces* — Li et al., HKUST(GZ) / Tsinghua, 2026 — [arXiv:2608.03451](https://arxiv.org/abs/2608.03451). Official KDD Cup 2026 Data Agents evaluation benchmark.
