# Memory Staleness in VLM Agents

*Depth — a safety failure mode of memory-augmented VLM agents that occurs when the world changes but the memory does not.*

**TL;DR:** Memory-augmented VLM agents rely on persistent spatial knowledge that silently goes stale as the environment changes. Sun & Zhang (2026) show three things on a dynamic FrozenLake testbed (1,800 detection runs, 12,000 navigation episodes): (1) text solvability does **not** imply visual grounding — models flag stale entries reliably from text yet score vision F1 from 0.887 down to 0.067 on the same grids; (2) trusting raw memory more than doubles agent death rate vs. no-memory baseline; (3) read-time filtering helps in text mode but not once visual auditing is unreliable.

**Prereqs:** [../multimodal/README.md](../multimodal/README.md), [../safety/README.md](../safety/README.md)
**Related:** [agent-harness.md](agent-harness.md) · [../safety/over-inference.md](../safety/over-inference.md)

---

## What it is

An agent maintains a persistent memory of spatial state — where obstacles are, which cell is safe. The environment then changes (an obstacle appears, a path flooded) *without* the agent's memory being updated. On the next visit, the agent must reconcile a confident memory claim ("cell X is safe") with a contradicting observation ("cell X is now a hole"). Memory staleness is the class of failures where the agent trusts the memory and acts on the outdated claim.

## How it works

The evaluation splits into two paired tasks on the *same* grids:

1. **Staleness detection.** Given a memory entry and current observation, does the model flag the entry as stale? Measured across text-only, image-only, and mixed inputs.
2. **Downstream navigation.** The agent plans a route using the (possibly stale) memory. Measure survival rate.

Key experimental variables: closed vs open-weight models (3 each), text vs image inputs, no-memory / raw-memory / audited-memory conditions, oracle vs model-provided staleness labels.

## Why it matters

- **Frames spatial memory as a safety property, not a quality property.** Deployed memory-augmented agents (ChatGPT Memory, Claude Projects, agent products with vector stores) currently have no guarantee against staleness-driven action.
- **Names visual grounding as *the* open problem.** The paper isolates the failure to a specific gap — models can *reason about* staleness from text but can't *see* it. That's a testable, addressable research target rather than a diffuse "memory is hard" complaint.
- **Auditing has a ceiling.** Even oracle stale labels don't close the survival gap on the current grid size — meaning read-time filtering is a partial defense, not a solution.

## Gotchas & tricks

- **"No memory" is a real baseline.** In their primary setting the agent *dies less* with no memory than with raw memory — a stark reminder that memory can be net-negative if not audited.
- **The gap is model-specific.** F1 range 0.887 → 0.067 shows some models can visually ground and some cannot. Choice of vision backbone matters as much as prompting.
- **Text-mode confidence is misleading.** Models that ace text-mode staleness detection are the *same* models that then ignore the image and confidently act on stale memory. Don't extrapolate text-mode benchmarks to vision-in-the-loop deployments.
- **Grid-size confound.** Their FrozenLake grids are small; larger observation spaces may recover some auditing headroom. The finding is a lower bound.

## Sources

- Paper: *When Memory Lies: An Empirical Study of Spatial Memory Staleness in VLM Agents* — Sun, Zhang, 2026 — [arXiv 2608.04574](https://arxiv.org/abs/2608.04574).
