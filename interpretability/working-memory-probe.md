# Working-memory probe (cumulative state tracking)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A lightweight probe for an LM's ability to keep a **running internal state across a sequence of in-context updates** without writing a scratchpad. A tunable *depth* parameter controls how many updates compound, so the probe stays hard as models improve. Introduced by the WMIP / WMF-AM paper (2603.27343). Tested on 28 models, correlates strongly with a 10-task agent battery, and the signal survives when the update content is swapped from arithmetic to permission/schedule/inventory tracking — pointing at cumulative tracking, not arithmetic, as the real bottleneck.

**Prereqs:** [README.md](README.md)
**Related:** [../evaluation/README.md](../evaluation/README.md)

---

## What it is

A single-query probe. The model is given a chain of "updates" — e.g. "add 3", "subtract 7", "double" — and must emit the final cumulative state. No chain-of-thought is allowed in the output: only the final value. Depth sets how many updates are chained.

The non-arithmetic variant uses permission grants, schedule changes, or inventory moves as updates — the surface content differs, but the cumulative-tracking demand is the same.

## How it works

Three design choices make the probe diagnostic:

1. **Depth parameter.** The number of compounded updates per query. Makes difficulty continuous and future-proof — raise depth when models saturate.
2. **No-scratchpad protocol.** Only the final answer is scored, so the probe measures *in-model* working memory, not externalized reasoning.
3. **Content swap.** Keep the depth and dependency structure identical; only change what the updates are "about". Rankings that are stable across arithmetic ↔ non-arithmetic content are evidence that the probe is isolating cumulative tracking.

Scoring is simple exact-match on the final state.

## Why it matters

- Small, cheap, infinitely rescalable — frontier benchmarks are saturating; this probe has a knob (depth) that can outpace models.
- Content invariance is unusual among probes: most "working memory" tests confound memory with the surface task (arithmetic, text manipulation). Splitting content from structure is a clean design.
- Correlates with agent-battery performance on 28 models, so it is a plausible early indicator for long-horizon agent competence.

## Gotchas & tricks

- Without the no-scratchpad rule, models silently CoT and the probe measures reasoning, not memory.
- Depth × model-size interaction is non-trivial: some small models match larger ones at low depth but collapse past depth ~8.
- Non-arithmetic content increases the chance of refusal or format slips; use strict output format and parse carefully.
- Don't use as a *standalone* capability eval — it's a diagnostic, not a leaderboard task.

## Sources

- Paper: *WMIP / WMF-AM: Probing LLM Working Memory via Depth-Parameterized Cumulative State Tracking* — Hou, Jiang, Li, Lin, Yamada (2026) — [arXiv:2603.27343](https://arxiv.org/abs/2603.27343)
