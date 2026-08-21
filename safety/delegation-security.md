# Delegation security (Agentic Principal Chain)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** LLM-based agents inherit permissions at session start and evaluate each request in isolation — a static, flat model of authority. The **Agentic Principal Chain (APC)** (Muruaga, 2026) makes authority *dynamic and chained*: every action is checked against a running record of who delegated what to whom, with six authorization checks per request. In evaluation APC dropped **AgentDojo exfiltration from 75–100% to 0%** across all four domains and blocked all **544 InjecAgent data-stealing cases**, at an 8.6–13.9 pp utility cost in some settings.

**Prereqs:** [_attacks.md](./_attacks.md)
**Related:** [../agents/README.md](../agents/README.md), [safety-case.md](./safety-case.md), [cot-monitoring.md](./cot-monitoring.md)

---

## What it is

A per-request authorization scheme for multi-agent / multi-tool LLM systems. Where a typical agent framework computes permissions once (at session start, from a static config) and treats every subsequent action as authorised if the caller has the right role, APC treats the *delegation history* itself as the authorization state.

The core object is the **principal chain**: an append-only record of who initiated the request, whom they delegated to, and under what constraints. A tool call is authorised only if the chain end-to-end satisfies six checks.

## How it works

Six per-request checks (schematically):

1. **Origin check.** Is there a genuine, non-forged initiator at the head of the chain?
2. **Delegation validity.** For every hop in the chain, did the delegator have authority to delegate *this specific capability*?
3. **Capability match.** Does the target tool's required capability sit within the delegated set at each hop?
4. **Scope match.** Are the request parameters within scope constraints attached at delegation time (data domain, output audience, side-effect class)?
5. **Freshness / expiry.** Has the delegation expired, been revoked, or been superseded?
6. **Combination guard.** Even if each hop is individually authorised, does the *combination* of prior actions in the session violate a policy (e.g. "read private data then write to external tool")?

Each request from an agent (whether to a tool, an API, or another agent) walks the six checks; a failure short-circuits the call.

## Why it matters

- **Closes a well-documented exfiltration hole.** AgentDojo and InjecAgent are benchmarks specifically designed to elicit data-stealing via prompt-injection-induced tool misuse; APC reduces attack success to zero across all four AgentDojo domains and blocks every one of 544 InjecAgent cases.
- **Substrate, not point-fix.** APC's checks target the *mechanism* (unbounded delegation) rather than any specific injection or jailbreak, so it composes with input filters and refusal training.
- **Prerequisite for real-world agent deployment.** Any agent handling private user data needs a principled delegation model; APC is one concrete answer.

## Gotchas & tricks

- **Utility hit is real.** 8.6–13.9 pp drops in some settings — many legitimate flows require multi-hop delegation across services and now hit the freshness / combination checks. Tuning per-workflow policies is part of deployment.
- **Chain integrity depends on the transport.** If an agent can forge a chain entry (e.g. impersonate a caller), the whole scheme collapses. APC assumes trusted signing at each hop.
- **Combination guard needs a policy language.** The paper sketches examples; production use needs an explicit, auditable policy for what combinations are forbidden — otherwise the check devolves into an ad-hoc allowlist.
- **Static delegations don't help.** Long-lived, wide-scope delegations recreate the flat model APC replaces; use short expiries and narrow scopes.
- **Not a jailbreak defence.** APC bounds *what* an agent can do with its permissions; it doesn't prevent the model from *deciding* to attack — pair with refusal training and CoT monitoring.

## Sources

- Paper: *Bounded Agents: Delegation Security for Multi-Agent AI Systems* — Muruaga, 2026 — [arXiv 2608.15888](https://arxiv.org/abs/2608.15888) — introduces the Agentic Principal Chain, the six-check scheme, and the AgentDojo / InjecAgent evaluations.
