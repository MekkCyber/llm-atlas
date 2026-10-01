# Context Language Models (CLM)
*Depth — LLMs that natively manage their own context by editing a file.*

**TL;DR:** A **CLM** treats its own context window as a *file* that the model can freely update: append, delete, rewrite, section, re-order. The model learns what to keep, drop, or compress — moving context management from external harness heuristics ("summarize every N turns") into intrinsic learned behavior. Co-designed with **Suffix Cache Reuse** for cheap serving. Introduced by Shao et al. (UW / Allen AI / Meta FAIR, 2026).

**Prereqs:** [README.md](./README.md), [../fundamentals/dca.md](../fundamentals/dca.md)
**Related:** [agent-harness.md](./agent-harness.md), [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

Long-horizon LLM agents hit context limits long before they hit capability limits. Classical workarounds live in the harness: summarize old turns, retrieve relevant chunks, truncate the oldest content. CLMs push the responsibility *into* the model: the context is exposed to the model as a mutable file, and the model learns a policy for editing that file as part of normal generation. In multi-agent setups, each agent is its own file.

## How it works

Two moves:

1. **Context-as-file I/O.** The model emits special actions (e.g. `edit(start, end, new_content)`) that rewrite spans of its own context in place. All subsequent generation sees the edited file.
2. **Training.** Zero-shot CLMs built on existing models already beat SOTA context-management strategies. Natural-language skills can be optimized via a standard skill-optimization loop (plus-up to +35.9 pt on held-out context-management tasks). An online RL method (group-relative-style objective akin to [GRPO](../post-training/grpo.md)) further improves Qwen3.5-9B on BrowseComp-Plus.

Serving uses **Suffix Cache Reuse**: when the model rewrites the tail of its file, only the modified suffix needs re-prefill — the shared prefix KV is reused. −35% server-side compute vs. standard SGLang at matched quality.

## Why it matters

Modern agent harnesses are increasingly dominated by heuristic context plumbing. CLMs collapse that plumbing into one learned action set, and empirically win on three stressful benchmarks:

| Benchmark | CLM vs. SOTA |
|---|---|
| BrowseComp-Plus | +11.4% accuracy, −21.5% FLOPs |
| 12-hour EdgeBench | +5% score, −59% FLOPs |
| 24-hour multi-repo swarm | +65% greater improvement at matched compute |
| BrowseComp-Plus (Qwen3.5-9B, online RL) | +47.6%, −12% FLOPs |

## Gotchas & tricks

- **Edits can delete critical state.** Early-training CLMs occasionally forget their own goals by over-aggressive compression. Keep a protected top-section ("goal frame") that the policy is penalized for touching.
- **KV reuse breaks on non-suffix edits.** If the model edits mid-file, the suffix after the edit must re-prefill. Reward functions that bias toward append-only edits trade a little capability for a lot of serving speed.
- **Zero-shot works, but it's brittle.** Natural-language skill prompts for context edits saturate fast; online RL is what unlocks the dramatic gains.
- **Shared with asynchronous agents.** CLMs and [asynchronous-agents.md](./asynchronous-agents.md) both expose the KV state as a first-class object the application controls. Pair cleanly in multi-process agent swarms.

## Sources

- Paper: *Context Language Models* — Shao, Shen, Yin, Li, Wang, Ivison, Poovendran, Lambert, Xiao, Lewis, Yih, Zettlemoyer, Koh — UW / Allen AI / Meta FAIR, 2026 — [arXiv:2609.37725](https://arxiv.org/abs/2609.37725).
