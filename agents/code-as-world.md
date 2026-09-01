# Code-as-World
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A physical-reasoning paradigm in which a VLM/agent iteratively discovers an **executable code representation** of a scene — object states, physical parameters, governing dynamics — then runs it to simulate outcomes, verify predictions, and revise. The final artifact is a runnable world model, not a natural-language chain of thought.

**Prereqs:** [README.md](README.md), [../post-training/reasoning/README.md](../post-training/reasoning/README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md) · [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md)

---

## What it is

VLMs can *describe* physical events but rarely have a stable internal model of the mechanisms behind them. Code-as-World grounds physical reasoning in an *external, executable representation* the model constructs itself — closing the loop between observation, symbolic representation, and simulation.

## How it works

An agentic loop with three roles:

1. **Propose a representation.** The VLM emits code that names the objects, their observable and hidden state (mass, friction, velocity), and the dynamics that govern them (equations, procedural rules, physics-engine calls).
2. **Execute.** The code runs — either as pure Python, or invoking a physics library / procedural sim — producing predictions the observed scene can be scored against.
3. **Revise on discrepancy.** Where predictions diverge from what the model actually observes (from images, video frames, or interaction feedback), the agent edits the representation: swap in a new dynamics term, adjust a parameter, split an object into parts.

Because the world model is code, **counterfactual interventions are cheap**: "what if we push harder / add friction / remove that wall?" is just a variable change followed by a rerun.

## Why it matters

- Gives an escape hatch from vibes-based physical reasoning: predictions are checkable, and disagreements localise to specific code lines.
- Supports **counterfactual reasoning** natively — a weak spot for pure VLM chain-of-thought.
- Fits neatly into agentic loops: proposing/editing code is exactly the loop other coding agents already run, so the same infrastructure carries over.

## Gotchas & tricks

- Representation search is unbounded. Without priors (physics library, canonical object taxonomy) the agent wastes rollouts inventing bespoke primitives.
- Simulation runtime is the tax. Rich physics sims are slow; use lightweight surrogates for iteration and only escalate to the expensive sim for final verification.
- The verification loop shares failure modes with any code-execution agent: silent errors, wrong-units bugs, and edge cases that pass the tests but miss the intent.

## Sources

- Paper: *Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning* — Wang et al., Tsinghua / NTU / MIT (large consortium), 2026 — [arxiv](https://arxiv.org/abs/2608.27549)
