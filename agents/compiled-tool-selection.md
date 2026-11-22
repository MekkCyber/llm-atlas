# Compiled Tool Selection (System 1 agents)

*Depth — replace generative tool-calling with a compiled multiple-choice question answered in one forward pass.*

**TL;DR:** Local LLM agents routinely hallucinate tool arguments. System 1 agents sidestep generation entirely: a deterministic compiler converts each tool schema and user request into a single multiple-choice question, and a lightweight decision model picks one option in a single forward pass. No text is generated, so no argument can be invented. On 1,952 requests over 157 biomedical tools, the best configuration hits 97.0% correct tool selection or correct refusal, with zero invented calls — vs. 5.5–19.7% invention rate for 18 local LLM-agent configurations.

**Prereqs:** [README.md](README.md)
**Related:** [../safety/README.md](../safety/README.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)

---

## What it is

The design point at the opposite end of free-form generative tool-calling: a *non-generative* agent. The model never emits text that will be parsed as a tool call; it classifies over a closed set of (tool, argument) choices precomputed from the tool schema plus the user request.

Suited to deployments where (a) the tool surface is bounded and known, (b) hallucinated arguments are unacceptable (clinical, legal, financial), (c) latency and privacy preclude cloud frontier models.

## How it works

Three stages:

1. **Compile.** For each tool in the schema registry and each user request, enumerate the candidate (tool, argument-binding) tuples that are type-valid. The result is an MCQ: stem = request, choices = tuples (plus "none / abstain").
2. **Decide.** Run the local decision model one forward pass on the MCQ and pick the highest-probability choice.
3. **Execute.** Call the chosen tool with the chosen arguments; or abstain if the "none" option wins.

Because the candidate set is deterministic and enumerated, hallucinated arguments are impossible by construction — the model can only pick among tuples it was given.

Compilation handles the combinatorics by pruning:
- **Type-filter** choices by argument type signature.
- **Value-filter** with request-conditioned heuristics (e.g., numeric ranges extracted from the stem).
- **Collapse** symmetric arguments where tool semantics are permutation-invariant.

## Why it matters

Shows that for latency- and privacy-constrained agent deployments, you can trade *generative flexibility* for *audit-grade reliability* and come out ahead on both correct-call rate and refusal correctness. Also a useful calibration baseline: the gap between a System 1 agent and an LLM agent quantifies the "freedom tax" the LLM is paying for generativity.

## Gotchas & tricks

- Scales with **number of tools × argument combinatorics** — large / open argument spaces (free-form strings, open numeric ranges) require aggressive pruning or hybrid designs.
- Loses expressive power for **multi-step planning** across tool chains; needs a planner wrapper (itself possibly a System 1 agent at each step) to compose.
- The decision model must be calibrated so the "none / abstain" option fires when nothing fits; standard classification-cross-entropy training plus a temperature sweep on a dev set is usually enough.
- Adding a new tool means recompiling the MCQ space, not fine-tuning the decision model — a deployment-time integration knob, not a training one.

## Sources

- Paper: *System 1 agents: a method for local, reliable and fast agentic AI* — Tzanis & Klontzas (University of Crete), 2026 — [arXiv:2611.21053](https://arxiv.org/abs/2611.21053).
