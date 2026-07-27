# Evolving-Intent Evaluation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A framework that transforms any single-turn benchmark into a multi-turn one in which the user's intent *evolves across turns* — incrementally revealed, revised, and sometimes redirected — while preserving the original evaluator. Reveals that strong static-benchmark performance does not transfer: models drop substantially across families when they have to *track* intent rather than answer it once.

**Prereqs:** [ifeval.md](./ifeval.md), [README.md](./README.md).
**Related:** [humaneval.md](./humaneval.md) · [mmlu.md](./mmlu.md) · [livecodebench.md](./livecodebench.md)

---

## What it is

A meta-evaluation protocol. Given an existing single-turn benchmark $B = \{(x_i, y_i^*)\}$ with a scorer $s(y, y_i^*)$, the framework scripts a **synthetic user** that:

- **Discloses intent in fragments** — reveals parts of $x_i$ across multiple turns.
- **Revises** — updates constraints as the conversation unfolds.
- **Redirects** — occasionally mid-course changes what's being asked.

At the end of the conversation, the model's final answer is scored with the *original* scorer $s$. No new annotation is required — the benchmark's ground truth is reused, only the delivery of the prompt changes.

## How it works

The synthetic user is itself an LLM prompted with:

- The full ground-truth question $x_i$.
- A revelation schedule (which parts to say when).
- Optional revision/redirection injections at scripted turns.

The model under test never sees the full $x_i$ in one go — it has to combine turn-by-turn disclosures into a coherent intent representation, ask clarifying questions if needed, and deliver a final answer that satisfies the accumulated (possibly self-inconsistent) intent.

This means any static benchmark (IFEval, HumanEval, MATH) can be "dynamised" without new labels. The paper transforms multiple benchmarks and reports the accuracy delta.

## Why it matters

Real users disclose intent iteratively; production agents live and die by how well they handle this. Yet all major LLM leaderboards are single-turn, fully-specified. Evolving-Intent is the first cheap, reusable protocol to measure the *tracking* capability at scale, and the paper's finding — that the gap is large and consistent across model families — suggests the static leaderboards have been over-selecting for a capability (one-shot answering) at the expense of a more important one (conversation-long intent modeling).

## Gotchas & tricks

- **Synthetic-user quality is a confound.** If the scripted user is inconsistent in ways the ground truth doesn't anticipate, the model may fail for reasons unrelated to its own tracking ability. Paper uses careful prompting; production use needs auditing.
- **Not a substitute for a real user study.** It probes tracking on tasks with a fixed reference answer; real dialogues have open-ended intents where "correct" is fuzzier.
- **Composable with tool-use benchmarks.** Nothing about the protocol assumes text-only tasks — an agent benchmark with tool calls can be dynamised the same way, and often shows even larger gaps.
- **Redirection ≠ noise.** A redirect that contradicts prior turns is a legitimate part of user behavior. Models that silently ignore contradictions and answer the "average" score better on the original score but worse on user satisfaction.

## Sources

- Paper: *LLMs Get Lost in Evolving User Intent* — Jihoon Tack, Philippe Laban, Jennifer Neville — Microsoft Research, 2026 — [arXiv:2607.20734](https://arxiv.org/abs/2607.20734)
- Code: [github.com/microsoft/evolving-intent](https://github.com/microsoft/evolving-intent)
