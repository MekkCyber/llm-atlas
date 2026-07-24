# VLM RLVR Environments (Trace)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RLVR has transformed LLM reasoning but is stuck for VLMs because there is no source of **broad, exactly verifiable, reproducible** visual training data. Trace is a taxonomy-guided environment that factorizes tasks into a **scene grammar** + **executable task program** — the same shared semantic state deterministically produces the image, prompt, typed answer, verifier state, and replayable instance trace. 1000 tasks across 11 visual domains; RLVR on 64k Trace instances lifts Qwen2.5-VL-7B by 4.06 pp macro-average across 24 external benchmarks.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [README.md](./README.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

An RLVR pipeline needs three things for every training instance: an input, a rollout, and a **cheap deterministic verifier**. Text-only RLVR gets all three easily — math answers, code test cases, format checkers. VLM RLVR breaks because you can't easily generate an image *and* a verifiable ground truth for a visual question in a way that's both broad (covers many task types) and cheap (no per-instance human labeling).

Trace's move: build environments around a **shared semantic state**. Every task begins with a semantic scene description; from that state, the environment deterministically renders the image, produces the prompt, computes the typed answer, initializes the verifier, and stores a replayable trace. The scene grammar controls what's possible; the executable task program says how to extract the answer.

## How it works

**Scene grammar.** A domain-specific language over visual concepts (objects, relations, transformations). Each grammar defines the space of valid semantic states for one visual domain (11 domains total: e.g. spatial arrangement, counting, chart reading, geometric reasoning).

**Executable task program.** A pure function of the semantic state that returns a typed answer. Because it's executable, the verifier is exact: no LLM-judge, no fuzzy matching.

**Rendering pipeline.** Given a semantic state, deterministic render → image + prompt. Any two runs with the same state produce identical inputs — reproducible.

**Instance trace.** For each rollout, the environment stores the state, the render, the model's response, and the verifier's decision. Traces are replayable, so a failure caught later can be re-run under the same conditions.

**RL loop.** Standard GRPO with rule-based rewards. 64k Trace instances → Qwen2.5-VL-7B trained end-to-end.

## Why it matters

VLM RLVR has been blocked by the **data problem**, not the algorithm. Trace shows that procedural, taxonomy-guided environments can produce broad-enough coverage to transfer out of distribution: 4.06 pp macro-average lift across 24 *external* benchmarks (not just Trace-style tasks) is the key result — the model generalizes beyond the generated distribution.

If this template holds, it's the analog for VLMs of what Codeforces-style verifiers became for code reasoning: a scalable source of RL-training data that doesn't require frontier-model judges or human labelers.

## Gotchas & tricks

- Coverage of the scene-grammar × task-program combinations is what determines OOD transfer. Too narrow and you overfit to the environment; too broad and per-task quality suffers.
- The "typed answer" constraint (integer, categorical, bounding box) is critical — free-text answers reintroduce fuzzy verification.
- 24-benchmark macro-average masks per-benchmark variance; expect some benchmarks to see minor regressions on style-heavy metrics.
- Not a replacement for real-image data. Trace is likely most powerful as a *complement* to natural-image SFT and preference data.

## Sources

- Paper: *Trace: A Taxonomy-Guided Environment for Multidomain Visual Reasoning* — 2026 — [arXiv:2607.19790](https://arxiv.org/abs/2607.19790)
- Related: DeepSeekMath (introduced GRPO — the RL algorithm used).
