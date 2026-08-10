# Agent memory compilation (Activity Frames)
*Depth — deterministic, zero-model compilation of raw screen-activity into agent memory.*

**TL;DR:** Agent memory today is either an LLM summary of past activity (drifty, non-cacheable) or a raw event log (too long for a prompt). Activity Frames sits in between: a deterministic pipeline compiles a screen-capture stream into typed, bounded episodes with **no model in the loop**, so the output is byte-identical, cacheable, and mechanically auditable — like a compiler's output rather than a summary.

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** *no depth files yet in this area*

---

## What it is

A memory representation and construction pipeline for computer-use agents. An "activity frame" is a typed, bounded episode carrying:
- `application` and `site`
- start / end / duration
- input volume (keystrokes, clicks)
- evidence pointers back to the raw capture rows

Frames are compiled from a local screen-activity capture stream by a deterministic segmenter. No LLM participates in memory construction; the LLM only *reads* the compiled block at query time.

## How it works

**Pipeline.**

```
raw capture rows  →  segmenter  →  typed activity frames  →  prompt-ready block
                     ^^^^^^^^^
                     deterministic
                     zero-model
```

**Segmentation.** Rule-based rules over (application-switch, focus-change, input-idle, URL-change) boundaries define frame edges. Every input row is either inside a frame or assigned to a boundary — no dropped events. Since the segmenter is a pure function of the input, running it twice on the same capture returns byte-identical output — the frames are cacheable at both the row and block levels.

**Prompt block.** The frames are serialized into a compact block (one line per frame with structured fields), with evidence pointers back to the raw rows for lookback queries. On the paper's corpus: 86× smaller than raw capture, compiled in 68 ms per day.

## Why it matters

- **Deterministic memory.** The output is a compiler's output, not a paraphrase — it caches, versions, and audits like code, which matters for agent products with governance requirements.
- **Cross-model portability.** A mid-tier model reading the compiled block matches a frontier model at 98.4% accuracy on day-questions, vs. 66–80% when both read an LLM summary of the same capture. Memory quality was the ceiling, not model choice.
- **Cheap inference.** No frontier-model calls to build memory; the LLM sees a prompt-ready block instead of re-deriving routines from raw activity.

## Gotchas & tricks

- Deterministic segmentation is only as good as the rule set — pathological capture streams (rapid tab switching, always-on-top windows) need per-app tuning.
- Evidence pointers back to raw rows are load-bearing for follow-up queries; the block alone answers "what did you do today?" but lookback queries need the pointers.
- The evaluation is single-user; multi-user or cross-device compilation is an open problem.

## Sources

- Paper: *Activity Frames: Deterministic Screen-Activity Compilation for Agent Memory and Replay* — 2026 — [arXiv:2608.05784](https://arxiv.org/abs/2608.05784)
