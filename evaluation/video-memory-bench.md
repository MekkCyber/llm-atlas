# Video memory benchmark (M³Eval)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** M³Eval (Huang et al., 2026) is the first benchmark for *memory* specifically in long-form video MLLMs — what models retain, how faithfully it is preserved, and how robust it is under interference. Tasks are organized against cognitive-psychology constructs (encoding, retention, retrieval, interference) rather than ad-hoc QA categories, so failures can be attributed to specific memory dimensions instead of being collapsed into one aggregate score.

**Prereqs:** [README.md](README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [streaming-spatial-mllm-bench.md](streaming-spatial-mllm-bench.md), [../agents/memory-self-supervision.md](../agents/memory-self-supervision.md)

---

## What it is

Long-context video models are now being shipped at hour-scale, and the bottleneck has moved from raw context length to *memory quality* — what the model actually retains across that long context. Existing video benchmarks score perception ("what color is the bag") and reasoning ("why did X do Y") but collapse retention, interference robustness, and retrieval into one aggregate number. M³Eval pulls them apart.

## How it works

Memory dimensions adapted from cognitive psychology:

1. **Encoding.** Was the information observed and registered in the first place?
2. **Retention.** Is it still available after time elapses (within the video)?
3. **Retrieval.** Can the model produce it on demand with the right cue?
4. **Interference robustness.** Does the model retain it under distractor content (visual interference, contradictory updates)?

Each video task is constructed so the failure mode it probes is well-defined. The same backbone can be compared on retention vs interference robustness in isolation — a separation existing video-QA benchmarks don't support.

## Why it matters

- **Diagnostic, not just leaderboard.** The four-dimension breakdown attributes failure to specific memory faculties. This is the substrate for principled work on memory-aware training.
- **Aligned with deployment.** Long-form video applications (lecture summarization, surveillance recall, multi-session AR assistants) are dominated by memory failures, not perception failures. The field needed a benchmark that says so.
- **Pairs naturally with memory training methods.** Work like [memory-self-supervision (MemTrain)](../agents/memory-self-supervision.md) gets a clean evaluation surface.

Current MLLMs show large gaps to human performance, with interference robustness and retention under distractors particularly weak.

## Gotchas & tricks

- **Construct validity is the central risk.** Cognitive-psychology constructs don't map perfectly to LLM behavior; the four-dimension partition should be treated as approximate.
- **Video distractors are expensive to construct.** Synthetic distractors (overlaid noise, scene cuts) bias the benchmark; in-scene distractors require careful video curation.
- **Long-form video evaluation is compute-heavy.** Each evaluation run is expensive; cache aggressively.
- **Aggregation matters.** Reporting one M³Eval score loses the diagnostic value. Always break out the four dimensions.

## Sources

- Paper: *M³Eval: Multi-Modal Memory Evaluation through Cognitively-Grounded Video Tasks* — Huang et al., 2026 — [arXiv:2606.05008](https://arxiv.org/abs/2606.05008).
- Affiliations: Peking University · University of Wisconsin–Madison.
