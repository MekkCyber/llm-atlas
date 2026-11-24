# Intent Contracts
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An **authorization interface** for tool-using LLM agents where the **planner holds no tools**. Each step, the planner emits a logged "contract" with a goal, a verbatim user quotation, and scope fields. A small trusted model validates the contract against the user's words and writes the actual tool call. Mirrors capability-based security: the large model cannot invoke tools directly.

**Prereqs:** [_scheming](../safety/_scheming.md)
**Related:** [cot-monitoring](../safety/cot-monitoring.md) · [safety-case](../safety/safety-case.md)

---

## What it is

Standard tool-using agents give one model both the planning authority and the tool-call authority — the same forward pass that decides what to do also emits the function invocation. This makes safety a function of the planner's jailbreak-resistance.

Intent Contracts splits the two authorities:

1. **Planner** — any large model. Can reason and compose plans. **Has no tools.** Its only output per step is a *contract*.
2. **Authorizer** — a small trusted model. Reads the contract and the chat history. Only if the contract is validated does it emit the actual tool call.

A contract has three fields:
- `goal`: what the step should accomplish, free-form.
- `verbatim_user_quote`: a literal substring of the user's message justifying this action.
- `scope`: structured fields (target object, allowed side effects, etc.).

---

## How it works

**Contract emission.** The planner is prompted / trained to produce contracts instead of tool calls. The quote field is enforced to be a substring of the user's recent turns; the scope field is a type-checked structured object.

**Validation.** The authorizer is a smaller model with narrow training: given the contract and the chat, decide if (a) the quote actually supports the goal, (b) the scope fields match what the quote references, (c) the intended tool call is within scope. Failures route back to the planner for revision.

**Tool-call authoring.** On validation, the authorizer — not the planner — writes the concrete tool call. The authorizer's tool-call surface is small and constrained.

**Logging.** Every contract is logged. Because quote and scope are explicit, post-hoc audit is straightforward.

---

## Why it matters

Current agent safety is monolithic: harden the planner against jailbreaks, bolt on action monitors, hope. Intent Contracts is **structurally different** — the planner simply **cannot** issue tool calls, so the attack surface moves to the authorizer, which has a much smaller surface and is cheap to re-train.

Compare with:
- Post-hoc monitors: still allow the planner to issue arbitrary calls; monitor races the call.
- Guarded planners: same model class as the attacker; alignment-brittle.
- Intent Contracts: capability-based separation. Even a fully compromised planner can only emit contracts.

---

## Gotchas & tricks

- **Ambiguous object references stay the dominant failure.** The verbatim quote may say "delete the file" without the authorizer knowing which file — AmPermBench results show this unresolved.
- **The contract improves task completion when the planner over-reaches** (pulls it back into scope) — but the AmPermBench numbers do **not** show reduced unauthorized actions. Capability separation is necessary but not sufficient; the scope-field language needs more precision.
- **Authorizer latency.** A small trusted model on every step is cheap but non-zero. Batching or local hosting matters for real deployments.

---

## Sources

- Paper: *Intent Contracts: Taking Tools Away from Agents* — Zhu, Chen, Wang, 2026 — [arXiv:2611.24025](https://arxiv.org/abs/2611.24025)
