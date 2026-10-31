# Interface Representation: MCP vs CLI (MICA & DART)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** When Model Context Protocol (MCP) and command-line (CLI) interfaces expose the *same semantic operations* through different model-facing representations, agents produce substantially different behaviors — even on identical capabilities with the same runtime. **MICA** (Matched Interface Comparison and Analysis) renders MCP and CLI from a shared *semantic action contract* and routes decoded actions through the same runtime, isolating representational effects. Across six models and three environments, **94.3% of 2,004 paired episodes diverge structurally**. **DART** (Decoupled Action Realization and Targeted Repair) preserves validated semantic commitments while repairing interface-specific realization, improving benchmark score by up to +9.0 points and cutting MCP token cost by up to −27.1k tokens/task.

**Prereqs:** none
**Related:** none

---

## What it is

MCP is being adopted as a de-facto agent interface. CLI has been the default for years. These expose different *representations* of the same operations: an MCP tool call `{"name": "read_file", "arguments": {"path": "x"}}` and a CLI command `cat x` are semantically identical actions through different model-facing surfaces.

MICA's hypothesis: **the representation is part of the decision environment, not surface.** Even at matched capabilities and matched runtime, the agent's choice of action, its failure pathways, and its recovery behavior depend on how operations are presented.

To test it cleanly, you need both wrappers to come from the *same* semantic action contract, and both wrappers to route decoded actions through the *same* runtime so differences cannot be attributed to downstream behavior.

## How it works

### MICA — the controlled probe

1. **Semantic action contract.** A single, representation-independent specification of operations: `{read_file(path), write_file(path, content), ...}`. The contract is written once.
2. **Rendered interfaces.** MCP and CLI are *renderings* of the contract into model-facing representations.
3. **Shared runtime.** Decoded actions on either interface route to the same executor. Behavioral differences cannot be attributed to runtime differences.
4. **Paired-trajectory analysis.** For each prompt, run the model under both interfaces and compare the trajectories.
5. **Matched-state probes.** Localize sensitivity to *capability presentation*, *realization*, and *feedback*.

### DART — the repair step

1. **Decouple action realization from semantic commitment.** Keep the model's semantic action choice (what to do) as the primary commitment; let the interface-specific realization (how to spell it) be fixable without changing the commitment.
2. **Targeted repair.** When the realization fails due to interface-specific quirks (e.g., MCP schema overhead, CLI quoting), repair the realization while preserving the committed semantic action.
3. **Result:** across six models and three environments, +≤9.0 pt improvement and ≤27.1k token reduction per MCP task.

### Headline finding

**94.3% of 2,004 paired episodes show structured divergence** — not noise, not stochastic sampling, but systematic representation-driven differences that propagate through subsequent interaction.

## Why it matters

- **Interface choice is a training-time variable, not a serving-time wrapper.** MCP vs CLI has been treated as a surface decoration; MICA shows it changes which actions the agent takes and how it recovers.
- **DART is a drop-in cost reducer for MCP agents.** The token-cost gap on MCP (verbose schemas) is a known deployment pain; targeted repair of realization bugs saves tens of thousands of tokens per task.
- **Reproducible comparisons.** The matched-contract, shared-runtime methodology is reusable for any future interface comparison (voice vs text, structured vs free-form, …).

## Gotchas & tricks

- **"Matched" is strong.** The contract has to be exhaustive — any operation that is reachable only through one wrapper breaks the comparison.
- **Representation effects compound.** The paper finds early differences propagate through subsequent interaction. One bad action early kicks the whole trajectory off; measuring per-step helps, but trajectory-level is where the gap shows.
- **MCP overhead is partly its own fault.** Token-cost reductions from DART are specific to MCP's schema-heavy realization. Different protocol choices could reduce the MICA gap without DART.
- **Six models, three environments.** Robust across tested range; generalization to coding IDEs and agentic frameworks beyond those three is plausible but not demonstrated.
- **Agent-training implication.** If the interface is a decision environment, training should include the realization layer, not freeze it as a wrapper.

## Sources

- Paper: *Matched Capabilities, Different Behaviors: A Controlled Study of MCP and CLI Interfaces for Language Agents* — Z. Shi, L. Pang, Y. Li, ModalityDance, 2026 — arXiv:2610.34298.
- Project: https://modalitydance.github.io/MICA/
- Code: https://github.com/ModalityDance/MICA
