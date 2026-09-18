# ProgramDistill
*Depth — a coding-agent benchmark that gives agents a working reference app and asks them to reimplement it into an incomplete copy.*

**TL;DR:** Most coding-agent benchmarks provide a natural-language spec or GitHub issue. ProgramDistill instead gives an agent a **fully functional reference application** and an incomplete copy, and measures whether the agent can *discover* the reference's behaviors through interaction and restore them into the copy. Behaviors are auto-mined via a `mine-craft-patch` pipeline that produces 1,975 replay-verified behaviors across 26 apps and 4,063 tasks, all without human annotation. Restoration depth is a controllable difficulty knob (1 → 8 features removed).

**Prereqs:** [humaneval](humaneval.md), [livecodebench](livecodebench.md)
**Related:** [codeforces-benchmark](codeforces-benchmark.md) · [ifeval](ifeval.md)

---

## What it is

A benchmark whose input is *working software*, not text. Given a reference app $A$ and a partial copy $A'$, the agent must interact with $A$, identify a set of behaviors, and add them back into $A'$ so its behavior matches $A$ on a replayable trace.

Behaviors are extracted by decomposing each app into features at multiple granularities. Each feature is associated with:
- A **replayable behavior trace** — a scripted interaction that exercises it.
- A **gold patch** — the diff that adds it to the incomplete copy.

The trace is the verifier: `A'-after-agent` must reproduce the trace's observed behavior on the reference app.

## How it works

The `mine-craft-patch` pipeline:
1. **Mine.** From a fully working repo, discover behaviors by exploring the app (button clicks, HTTP calls, page loads) and recording input/output pairs.
2. **Craft.** Cluster and factorize behaviors into feature units at multiple granularities (single-function → multi-file feature).
3. **Patch.** For each feature, compute the diff that removes it, then use the diff as the gold patch and the pre-removal trace as the verifier.

**Restoration depth $d$** = number of features removed from the copy before handing it to the agent. Higher $d$ compounds cross-feature interactions and forces the agent to plan.

Applied to 26 apps: **1,975 verified behaviors, 4,063 tasks**, zero human annotation.

## Why it matters

- **Escapes SWE-bench saturation.** Frontier agents (GPT-6 Astra, Claude Opus 5) score 49.2% / 28.8% on full-application reconstruction — plenty of headroom, and the depth knob keeps it that way.
- **Cheap, scalable, controllable.** No human labels; any working app becomes a benchmark. Difficulty is tunable per task.
- **Natural curriculum for RL post-training.** The paper explicitly proposes this: sample restoration depth per training episode.
- **Clean failure signal.** Depth-1 success is near-perfect for both agents (100% Astra, 96% Opus 5); depth-8 collapses to 64% / 32%. Isolates *cross-feature reasoning*, not one-shot codegen.

## Gotchas & tricks

- **"Behavior" is trace-defined, not spec-defined.** A patch that reproduces the trace but violates non-observed contracts passes the verifier — expect gaming as agents get stronger.
- **Reference apps are the ceiling.** Diversity of apps limits diversity of behaviors. The pipeline generalizes, but 26 apps is still narrow.
- **Depth ≠ difficulty monotonically.** Some depth-8 tasks are easier than depth-4 tasks with tangled feature dependencies. Report per-depth *distributions*, not just means.

## Sources

- Paper: *ProgramDistill: From Interactive Web Apps to Verifiable Reference-Guided SWE Tasks* — Kim, Kim, Kim, Pereira, Côté, Sordoni, Yuan, Shi, Microsoft / MSR Montréal, 2026, arXiv 2609.18805.
