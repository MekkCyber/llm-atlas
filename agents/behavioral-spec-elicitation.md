# Behavioral Specification Elicitation

*Depth — a two-agent scaffold that separates probing/spec-writing from code synthesis in from-scratch program construction.*

**TL;DR:** Single-loop coding agents conflate three activities: reading docs, probing the reference binary, and writing the code. On from-scratch program construction (ProgramBench, <1% frontier baseline) this causes under-probing and lets early misinterpretations propagate. SpecFirst forces a two-stage scaffold: a **spec agent** probes the binary + reads the docs and emits a structured **behavioral specification**; a **code-synthesis agent** implements against the spec. Same models, different scaffold — +6.9 to +21.3 pp on ProgramBench across four models.

**Prereqs:** *(none — a scaffold pattern applicable to any coding-agent stack)*
**Related:** [README.md](./README.md), [source-free-env-construction.md](./source-free-env-construction.md)

---

## What it is

A scaffold pattern for LLM coding agents in the **from-scratch program construction** setting: the only inputs are natural-language documentation and an execute-only reference binary that acts as a behavioral oracle. The pattern factors the agent into two sequential phases:

- **Spec elicitation.** A dedicated agent probes the binary (calls it with varied inputs, observes outputs / exit codes / side-effects), reads the docs, and emits a structured behavioral specification — think requirements document, not source code.
- **Code synthesis.** A second agent uses the spec as its stable reference and writes an implementation. It may still call the binary, but the spec is what it commits to.

## How it works

- **Spec agent role.** Its output is *not* code. It's a structured artifact — inputs, outputs, error modes, ambiguous edge cases resolved by probing, invariants. Format is up to the implementer; the paper uses a structured template.
- **Behavioral probing.** The spec agent's tool budget is spent on running the binary with diverse inputs to disambiguate documentation. This is the "requirements engineering" step, imported from classical SE.
- **Handoff.** The code-synthesis agent starts with a fresh context window primed with the spec. It doesn't re-probe (much); the spec is authoritative.
- **Model-agnostic.** Works across four evaluated models spanning two families and an order-of-magnitude of capability. The gains are structural, not model-specific.

## Why it matters

- **Documentation ambiguity is resolved once, in a stable place.** A single-loop agent re-derives behavior from probing on every attempt and forgets it under context drift. A written spec doesn't.
- **Inspectable handoff artifact.** The spec is a natural review point — humans or other agents can validate the spec before code is committed. Aligns with the general "explicit intermediate artifacts" pattern in agent design.
- **Points at SE-lifecycle-shaped scaffolds.** SpecFirst is one instance of a broader argument: coding agents should mirror phases of the software lifecycle (requirements → design → implementation → test) rather than compress them into a single planner-executor loop.

Reported results across four models on all 200 ProgramBench instances:
- +6.9% to +21.3% test-pass improvement.
- +9.4% to +18.5% binary exploration coverage.
- All statistically significant.
- Behavioral analysis shows code synthesis starts *earlier* and is more sustained once spec-first.

## Gotchas & tricks

- **Spec quality is the ceiling.** A bad spec propagates; the code-synthesis agent trusts it. Invest in the spec agent's tool budget and prompt.
- **Format choice matters.** Free-form specs regress toward re-derived paraphrases. Use a structured template with slots the spec agent must fill.
- **Two agents ≠ two models required.** The paper uses the same model in both roles; the *scaffold* — not the model split — is the source of the gain.
- **Adds one round-trip of latency.** For interactive coding assistants where latency matters, spec-first is a batch-processing pattern; not the shape you want for line-by-line completion.
- **Companion to [source-free-env-construction](./source-free-env-construction.md).** SpecFirst is the *scaffold* side of ProgramBench-flavored progress; the *training-data* side is source-free env construction. Both from the same team.

## Sources

- Paper: *SpecFirst: Behavioral Specification Elicitation as a First-Class Step in Agent-Based Program Synthesis from Scratch* — Chen et al., 2026 — introduces the two-stage scaffold and benchmarks it on ProgramBench. See [../daily-papers/2026-07-30.md](../daily-papers/2026-07-30.md).
- Related: [source-free-env-construction.md](./source-free-env-construction.md) (companion MindForge paper from the same team).
