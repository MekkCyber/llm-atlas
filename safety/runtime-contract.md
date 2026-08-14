# Runtime Contract
*Depth — safety guarantees for deployed agents expressed as runtime obligations plus evidence chains, not just training-time properties.*

**TL;DR:** A runtime contract binds an agent's every action to two things: **preventive gates** (sandboxes, permission checks, filters that block risky ops before they land) and **evidential requirements** (a mechanically checkable proof that the action was authorized and completed correctly). The unit of safety is the *trajectory-with-checkable-evidence*, not the model. Reframes agent safety from "the aligned model won't do it" to "the runtime observed and can prove what happened."

**Prereqs:** [_attacks](_attacks.md), [_agent-attacks](_agent-attacks.md)
**Related:** [safety-case](safety-case.md), [cot-monitoring](cot-monitoring.md), [_scheming](_scheming.md), [indirect-prompt-injection](indirect-prompt-injection.md)

---

## What it is

A specification the *deployment* enforces, independent of the model's disposition. Two components:

1. **Preventive layer.** Sandboxes, permission gates, allow-lists, content filters, per-tool policy checks — enforced *before* the action commits.
2. **Evidential layer.** For each action the agent takes, the runtime records a typed *evidence chain* — inputs, tool schemas, output validation, authorization tokens — that can be checked mechanically against the contract. An action without a well-formed evidence chain is treated as a policy violation, whatever the model intended.

Formalized as an *Agent Trajectory Schema and Evidence Chain* by Ng et al. 2026 after an audit of 52 documented safety incidents.

## How it works

The pattern:

```
model proposes action
    → runtime resolves it against the trajectory schema
    → preventive gates evaluate (permission, sandbox, filters)  → deny or continue
    → action executes in the sandbox
    → runtime records typed evidence (inputs, outputs, auth tokens, validation)
    → next step reads the evidence, not the raw tool output
```

Every step is checkable *after the fact* by a monitor that never runs the model. This is what makes the contract enforceable even when the model is compromised, jailbroken, or scheming.

## Why it matters

- **Aligns the guarantee with what deployments can actually verify.** Model dispositions are hard to certify; runtime behaviors are logs.
- **Compatible with closed-weight models.** No training-side access required.
- **A concrete defense against [indirect-prompt-injection](indirect-prompt-injection.md).** Injected instructions cannot manufacture the required authorization tokens or evidence, so downstream actions fail closed.
- **Ties into [safety-case](safety-case.md).** Evidence chains are exactly the kind of structured argument a deployment-time safety case wants.
- **Makes trajectory-level monitoring first-class.** A natural home for [cot-monitoring](cot-monitoring.md) and post-hoc audits.

## Gotchas & tricks

- **Evidence chains rot silently.** If the schema doesn't evolve with the tool surface, the runtime accepts unauthenticated actions that pass old checks. Version the schema; fail-closed on unknown fields.
- **Sandboxes leak.** Preventive gates that rely on the sandbox for isolation must assume the sandbox is escapable — pair with post-hoc audits that catch escape signatures.
- **Model-generated evidence is not evidence.** If the model itself writes the "authorization" string, an attacker who controls the model can forge it. Auth tokens must be minted outside the model's context.
- **Bias toward under-approximation.** Runtime contracts naturally over-block. Ship a graceful "why was this denied" surface — otherwise operators disable the gate at first friction.
- **Publication imbalance.** The paper notes training-time safety research vastly outpaces runtime — the field's default reflex is still "train it out," which is not sufficient for agents.

## Sources

- Paper: *Agent Safety Should Be a Runtime Contract* — Ng, Han, Zhang, Wang, 2026 — [arXiv:2608.11274](https://arxiv.org/abs/2608.11274) — canonical framing and the trajectory-schema + evidence-chain formalism.
- Paper: *Safety Cases: How to Justify the Safety of Advanced AI Systems* — Clymer et al., 2024 — the structured-argument framework runtime contracts naturally slot into.
