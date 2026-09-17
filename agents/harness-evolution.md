# Harness Evolution
*Depth — RSI branch that edits the agent scaffold while keeping the model frozen.*

**TL;DR:** Harness evolution is the RSI mode in which only the *scaffold around* an LLM — prompts, tools, routing rules, context-management policies — is updated from trajectories, while the model weights stay fixed. It's the cheapest RSI variant (no gradient step, no GPU cluster) and works surprisingly well on long-horizon coding and terminal tasks, but its ceiling is bounded by the frozen model's capability.

**Prereqs:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)
**Related:** [_recursive-self-improvement](_recursive-self-improvement.md), [modular-harness](modular-harness.md), [recursive-in-recursive-si](recursive-in-recursive-si.md)

---

## What it is

A closed loop that runs the agent on tasks, extracts what went wrong or right from the resulting trajectories, and rewrites the *harness* — the collection of prompts, tool definitions, memory policies, and orchestration code around the model — to reduce recurrent failure modes. No model training occurs. In practice it looks like a self-improving prompt library plus tool wrappers.

## How it works

1. **Trajectory collection.** Run the current agent on a training set of tasks; log full transcripts (thoughts, tool calls, observations, outcomes).
2. **Deficiency attribution.** Cluster failures by symptom. A common trick (from ModularRSI, 2026) is to *contrast* successful and failed trajectories on the same task so that instance-specific reasoning cancels and only recurring scaffold problems remain.
3. **Harness edit proposal.** A meta-LM proposes local edits (a new prompt clause, a stricter tool schema, a re-orderable tool list) targeted at each deficiency class.
4. **Validation.** Re-run on a held-out task pool; accept edits that raise pass rate, roll back the rest.
5. **Integration.** Merge accepted edits into the harness and repeat.

The key discipline is separating the *evolution set* from the *evaluation benchmark* — otherwise the harness overfits to the benchmark rather than gaining a reusable capability.

## Why it matters

Harness evolution is a fast, cheap, deploy-time-compatible way to raise agent performance without touching weights. It's the only RSI variant available when the model is closed-weights or fine-tuning is banned by policy. Combined with an untouched foundation model, an evolved harness typically transfers across model families — a rare property in post-training.

## Gotchas & tricks

- **Benchmark leakage.** Evolving on the eval benchmark yields specificity, not capability. Use *benchmark-disjoint* task pools (ModularRSI curates ~2k).
- **Whole-harness edits entangle unrelated modules.** Split the harness into functional modules (Agent Loop, Tool Use, Observation Management, Context Management, Task Completion Detection) and evolve each in a restricted scope, then integrate.
- **Single-trajectory edits overfit.** Aggregate deficiencies across many failures before proposing an edit.
- **Model-agnostic wins are the exception.** Most harness edits still bake in the current model's failure profile; validate on multiple backbones before claiming transfer.
- **The ceiling is the model.** Once the frozen model's ability to follow a scaffold saturates, further harness edits stop helping — that's when to switch to coupled RSI.

## Sources

- Paper: *ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement* — Wu et al., 2026 — [arXiv:2609.14857](https://arxiv.org/abs/2609.14857).
- Paper: *ScienceBuddy: Recursive-in-Recursive Self-Improvement for Interactive Scientific Agents* — Xue et al., 2026 — [arXiv:2609.17523](https://arxiv.org/abs/2609.17523).
