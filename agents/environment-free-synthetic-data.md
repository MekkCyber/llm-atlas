# Environment-free Synthetic Data for API-Calling Agents

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Training API-calling agents demands large trajectory datasets, and building fully implemented environments to collect them is the bottleneck. This pipeline skips the environment entirely: given only the API specification, one LLM acts as **task generator**, another as **environment simulator** (producing plausible API responses to any call), a teacher agent solves the tasks against the simulator, and a **judge LLM** filters the trajectories. Fine-tuning on the survivors improves agent performance on real environments (AppWorld, OfficeBench) without ever exercising a real API.

**Prereqs:** [_data-curation.md](../data/_data-curation.md), [rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [quality-filtering.md](../data/quality-filtering.md) · [_post-training.md](../post-training/_post-training.md)

---

## What it is

A three-LLM synthetic-trajectory pipeline for tool-use / API-calling training data. All three roles can be the same base model or a stronger proprietary model; the essential structure is the role separation and the schema-conditioned simulation.

## How it works

Given an API specification (endpoint names, arguments, return schemas):

1. **Task generator.** An LLM proposes plausible user goals expressible via the API — spanning single-call, multi-call, and error-recovery scenarios.
2. **Teacher agent.** A second LLM (usually a strong instruction-tuned model) attempts each task, emitting API calls one at a time.
3. **Environment simulator.** For each teacher call, a third LLM — conditioned on the API spec plus a growing "world state" cache — produces the API response, including realistic errors and edge cases. This is the environment.
4. **Judge.** A judge LLM evaluates completed trajectories against the task and filters low-quality ones.
5. **Train.** The surviving trajectories are used as SFT data for the student agent.

The world-state cache is critical: repeated calls must return consistent responses, or the training data teaches the student that the environment is nondeterministic in weird ways.

## Why it matters

Real-environment data collection has been a moat for closed labs training tool-use models — you need running services, sandboxes, quota, cleanup. This pipeline turns the moat into a spec + three LLM calls. Apple publishing it is notable given their on-device-agent trajectory. The generality of the recipe (works for any API-shaped tool: HTTP, MCP, function-calling) makes it a natural default for the next generation of open agent training data.

## Gotchas & tricks

- Simulator consistency is the hardest bit. Without a state cache, repeated GETs and stateful POST/DELETE sequences drift, and the student learns garbage.
- The judge is a single point of failure — a lax judge floods the dataset with bad trajectories. Pairwise judging or multiple-judge voting improves robustness.
- Fine-tuning on synthetic API responses doesn't teach *real API quirks* (rate limits, actual error messages). A small real-environment fine-tune on top typically closes the gap.

## Sources

- Paper: *Environment-free Synthetic Data Generation for API-Calling Agents* — Seanie Lee, Sanjoy Chowdhury, Chao Jiang, Cheng-Yu Hsieh, Ting-Yao Hu, Alexander T. Toshev, Oncel Tuzel, Raviteja Vemulapalli (Apple), 2026 — [arXiv:2607.16900](https://arxiv.org/abs/2607.16900) · [HF](https://huggingface.co/papers/2607.16900)
