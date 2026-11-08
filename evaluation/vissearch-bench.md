# VisSearch-Bench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for multimodal agents on long-horizon visual search — tasks that combine image understanding with external information retrieval and multi-step reasoning. Introduced by Ant Group (2026). Headroom is enormous: the best frontier system fully solves only **26.47%** of tasks.

**Prereqs:** [humaneval](humaneval.md)
**Related:** [mmlu](mmlu.md), [codeforces-benchmark](codeforces-benchmark.md)

---

## What it is

Current multimodal benchmarks either saturate (VQA, LVLM-bench) or test a single capability in isolation. VisSearch-Bench specifically targets the *loop* that production multimodal agents run: look at an image, decide what external information is needed, query, read results, combine with the image, and answer.

Each task is constructed so that (a) the answer cannot be inferred from the image alone, (b) the answer cannot be retrieved from text alone, and (c) the agent must take multiple search/read steps.

## How it works

**Task construction:**
- Start from an image with a specific entity or scene.
- Compose a question whose answer requires identifying the entity visually, then looking up facts about it that are not visually apparent (history, specifications, relationships to other entities).
- Validate that no single search query can return the full answer.

**Evaluation:**
- End-to-end task success (binary) — not per-step correctness.
- Agents run under a standardised harness (browser + VLM backbone + a retrieval tool).
- Time/cost budgets are logged but not scoring criteria.

## Why it matters

Public benchmarks like LVLM-eHub and MMBench are saturating; frontier multimodal LLMs get ≥90% on them, which underpredicts failures in deployed agents. VisSearch-Bench reopens a large gap specifically on the vision × retrieval × reasoning chain, which is the weakest part of current multimodal agent stacks.

Error analysis in the paper isolates two dominant failure modes: (1) **grounding failures** where the agent misidentifies the image entity and then searches for the wrong thing; (2) **premature answer commitment** where the agent answers before reading enough retrieved context.

## Gotchas & tricks

- **Not a static benchmark.** Because the tasks depend on live web retrieval, scores drift as the open web changes. The paper pins a snapshot; evaluators should report both snapshot-pinned and live scores.
- **Harness sensitivity.** A different retrieval tool (Google vs Bing vs a specialised crawler) changes scores substantially. Report harness.
- **No contamination guarantee** at frontier scale; some entities may be in training data. The 26.47% ceiling suggests contamination isn't the limiting factor today.

## Sources

- Paper: VisSearch-Bench: Evaluating Long Horizon Visual Search in Multimodal Agents — Wu et al., 2026 — [arXiv:2611.04894](https://arxiv.org/abs/2611.04894)
