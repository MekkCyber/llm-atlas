# Agent privacy leakage (acquisition-stage)
*Depth — auditing what an agent pulls into context, not just what it outputs.*

**TL;DR:** Existing agent-privacy benchmarks audit what an agent's *output* or outgoing action discloses. That misses the earlier stage where sensitive data first enters the agent's context via tool calls. Over-acquired data is one prompt-injection or careless downstream action away from leaking. Acquisition-stage privacy leakage frames the context window itself as the privacy surface, splitting audit into (1) Acquisition Inspection over the tool-call trajectory and (2) Probe Elicitation via follow-up queries against the retained context.

**Prereqs:** [../agents/README.md](../agents/README.md), [README.md](README.md)
**Related:** [_attacks.md](_attacks.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

An LLM-based agent completes multi-step tasks by invoking external tools (search, database, APIs). Each tool call may return more information than the current step actually needs, and that data lands in the agent's context. Even if the agent never surfaces it, the data is exposed to every subsequent step, tool call, and prompt-injection attempt.

Agent-privacy leakage in the *acquisition* stage is any case where the agent pulls sensitive information into its context beyond the scope of the assigned task.

## How it works

The two-stage audit:

**Acquisition Inspection.** For each task trajectory:
1. Trace every tool call the agent makes and the data each call returns.
2. Label per-call: was this data within the task's stated scope? Sensitive attributes (PII, financial, medical, credentials) flagged.
3. A trajectory is *over-acquiring* if any call pulled sensitive out-of-scope data.

**Probe Elicitation.** After the trajectory concludes, issue a follow-up probe:
1. Send a case-specific probe query targeting a sensitive attribute the trajectory acquired.
2. Measure whether the agent discloses that attribute from its retained context.
3. Disclosure = evidence that the acquired data is *reachable* by future prompts, not just held.

The evaluated failure modes:
- **Acquisition without disclosure:** data is in context but not surfaced yet — a latent leakage risk.
- **Acquisition with disclosure:** data both pulled and later revealed.

## Why it matters

- **Reframes the privacy surface.** Prior benchmarks treat the model's *output* as the surface; this reframes it as the *context window*.
- **Empirically widespread.** PrivacyPeek finds unnecessary sensitive-data acquisition across 10 agents / 4 model families.
- **Correlates with capability.** More task-competent agents leak *more* at acquisition stage — the better they use tools, the more they pull.
- **Prompt-level defenses barely help.** System-prompt-only mitigations reduce only a small fraction; most leakage is unmitigated. Motivates tool-side / architecture-side interventions (scoped tool wrappers, per-call minimization).

## Gotchas & tricks

- **Scope labeling is judgment-heavy.** What counts as "out of scope" depends on task interpretation; use per-domain annotators.
- **Probes must be strong.** Weak probes miss disclosure that a stronger probe would surface; report probe strength alongside disclosure rate.
- **Over-acquisition ≠ malicious.** Often the agent pulls extra data because tools return coarse-grained results (e.g. a whole record instead of one field). Fixing tools may be as impactful as fixing the agent.
- **Retention is task-length dependent.** Long-horizon agents retain acquired data for many steps; disclosure risk grows with horizon.
- **Complementary to output-level jailbreak evals** (see [_attacks.md](_attacks.md)) — same threat model, different surface.

## Sources

- Paper: *PrivacyPeek: Auditing What LLM-Based Agents Acquire, Not Just What They Say* — Zhang, Han, Guo et al., 2026 — arXiv:2606.00152 — 1,182 cases across 7 acquisition behaviors and 16 domains; 10 agents across 4 model families.
