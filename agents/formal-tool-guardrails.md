# Formal Tool Guardrails
*Depth — runtime admissibility checks on agent tool calls, discharged as formal proofs.*

**TL;DR:** A tool-using agent's safety can be enforced by checking each proposed tool call against a policy written in a formal proof system (Lean 4 in LeanGuard, 2026) before the call fires. Rejected calls can be returned to the model with the violated rule as feedback, or auto-repaired by the proof system searching for a nearby admissible call. Unlike LLM-judge guardrails, a formal check cannot be prompt-engineered away: it either discharges the proof or it doesn't.

**Prereqs:** [../safety/_attacks.md](../safety/_attacks.md) · [../safety/safety-case.md](../safety/safety-case.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md) · [../safety/_scheming.md](../safety/_scheming.md)

---

## What it is

A **formal tool guardrail** is a runtime layer between an agent's tool-call proposal and the tool's execution. For each proposed call `call(tool, args)` in context `state`:
1. Encode the agent's current authorization and world state as a context.
2. Ask the formal system to discharge `admissible(tool, args, state)`.
3. If the proof goes through, execute the call. If not, deny.

The admissibility predicate is a formal definition — in LeanGuard's case, a Lean 4 `def` over the tool call's arguments and the agent's current state — written once per deployed policy. Typical rules: the agent may not call `send_email` with a recipient outside the user's domain; the agent may not call `write_file` outside the sandbox root; the agent must have an explicit user confirmation on record before any call tagged as destructive.

## How it works

- **Policy as proof obligation.** The admissible-action predicate is a total definition in the proof language. Every tool call must produce a proof witness; the runtime discharges this by calling the proof assistant with the policy and the call as inputs.
- **Two modes of recovery from a denial:**
  - **Feedback mode** — the violated rule is formatted as a message and returned to the model; the model proposes a different call.
  - **Auto-repair mode** — the proof system does a local search over nearby admissible calls (same tool, adjusted arguments) and returns the first one whose proof discharges. The agent gets the repaired call directly.
- **World state** — the runtime maintains a typed representation of the agent's sandbox, available credentials, open files, past tool calls. The policy is parameterized on this state.

In LeanGuard, the proof engine is Lean 4 and the sandbox has a Python/Docker twin. Proof discharges take tens of milliseconds for the policies tested, dominated by the Lean kernel's typechecking.

## Why it matters

- **Can't be social-engineered.** An LLM judge can be jailbroken; a Lean proof cannot. For high-stakes domains (finance, healthcare, infrastructure) this is the one guarantee the LLM safety stack currently lacks.
- **Composes with LLM guardrails.** Formal checks catch structural violations (arguments, scope); LLM judges catch semantic violations (tone, intent). Different layers, different attack surfaces.
- **Auditable.** Every denial has an explicit proof-system trace: this rule rejected this call because of this fact in the state.

## Gotchas & tricks

- **Policy coverage gap.** A policy that only partially specifies admissibility has a false-reject surface — admissible-in-spirit calls that fail the proof. Monitor that gap or legitimate workflows break.
- **State representation lag.** The proof is only as good as the modelled world state; if the sandbox state diverges from reality (external side-effects not captured), the proof is lying.
- **Slow policies.** Deeply nested `def`s with many obligations can push the proof-discharge time past interactive latency. Prefer flat admissibility definitions with precomputed state predicates.
- **Writing policies is expensive.** The upfront cost is a formal-methods engineer hours per policy. Scale only to a handful of high-stakes tools.

## Sources

- Paper: *LeanGuard: Lean-Verified Guardrails for Tool-Using Agents* — Sankar et al., Case Western Reserve / UT Austin, 2026 — [arXiv:2611.02332](https://arxiv.org/abs/2611.02332).
- Benchmark: τ²-bench, DTap — used in the paper to show unguarded agents execute policy-violating calls.
