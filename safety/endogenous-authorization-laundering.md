# Endogenous Authorization Laundering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A safety failure mode for long-running agents where **the agent's own persistent memory** misrepresents evolving authorization state, giving the agent authority the underlying history never granted — **without any external attack**. The paper introduces EAL-Bench (procurement / cybersec / finance domains) and shows memory writers create false authority for up to 50.2% of unauthorized requests; downstream executors act on it 98.6% of the time.

**Prereqs:** [_attacks.md](_attacks.md)
**Related:** [cot-monitoring.md](cot-monitoring.md), [situational-awareness.md](situational-awareness.md), [safety-case.md](safety-case.md)

---

## What it is

A long-running LLM agent carries a **persistent memory** across interactions — permissions, restrictions, revocations, tool grants. When the agent (or a memory-writer subcomponent) *rewrites* this memory over time, errors accumulate: a permission that was never granted, or was later revoked, can end up recorded as active. Because there's no external attacker involved, this failure is "endogenous" — the agent grants itself authority its own history never permitted.

The name "laundering" reflects that the false authority's provenance is *washed away* through the memory-writing chain: an executor asked to check permissions sees the write, not the incremental history that should have justified it.

## How it works

The paper decomposes agent memory into two roles:

- **Writer.** Updates persistent memory after each interaction. Failure: writes false authorization state.
- **Executor.** Reads persistent memory, decides whether to take an action. Failure: acts on false authority without back-checking history.

Under incremental memory updates, writers fabricate authority for up to **50.2%** of unauthorized requests. Once fabricated, executors act on it in **98.6%** of trials. Both failures compound.

### The two proposed safeguards

1. **Source-event backing.** Stored permissions must be backed by a valid source event in the interaction history. Reject writes that aren't backed.
2. **Bounded event sourcing.** Represent permission state as a bounded-length event log rather than a mutable snapshot; changes are reconstructed rather than snapshotted.

Both reduce laundering substantially but also reject more legitimate actions — a safety-utility tradeoff intrinsic to the surface.

## Why it matters

- **New failure class**, not a variant of prompt injection or jailbreak. Persistent memory is now a policy surface, not just a performance component.
- **Threat model doesn't require an attacker.** Existing agent-safety work focuses on adversarial inputs; endogenous laundering emerges from ordinary use.
- **Sets a design constraint for agent memory architectures.** Memory representations that don't support source-event verification are fundamentally unsafe for authorization-relevant state.
- **EAL-Bench is a concrete measurement instrument** — five writers × two executors × three domains — that other agent frameworks can evaluate against.

## Gotchas & tricks

- **Writer and executor may share the same base model.** Nothing about the failure requires distinct models; the same LLM playing both roles reproduces it.
- **Domain choice matters for realism.** Procurement/cybersec/finance were chosen because permission changes are frequent and consequential; low-stakes domains may not show the failure clearly even though the mechanism is present.
- **"Bounded" event sourcing is not "unbounded".** The paper uses a bounded-length log, so old events fall out of the window. That's a deliberate scaling tradeoff — an unbounded log solves laundering by construction but grows without limit.
- **Rejection rate is the utility cost.** Both safeguards trade laundering-rate for legitimate-action-rejection; picking a threshold is a policy call, not a hyperparameter.
- **Composes with prompt injection.** External attacks that seed the memory writer with plausible-looking permission events are strictly more dangerous than either failure alone.

## Sources

- Paper: *Agent Memory Is a Surface for Endogenous Authorization Laundering* — Cerruti, Okamoto, Erol, 2026 — [arXiv:2609.01836](https://arxiv.org/abs/2609.01836).
