# Policy Workflow Graphs (PolicyGuide)
*Depth — compile natural-language organizational policy into a workflow graph and enforce compliance via a proactive per-turn verifier.*

**TL;DR:** Customer-service LLM agents fail organizational policy in two shapes: **forbidden actions** (grant an ineligible change) and **omitted procedural requirements** (skip identification or confirmation). Action-local guardrails catch (1) but not (2); workflow-following systems target completion, not compliance. **PolicyGuide** compiles each domain policy into a **workflow graph**, keeps state persisted across turns, and invokes a **proactive verifier at user-turn boundaries** that reconciles open requests and returns step-specific remediation on a policy-compliant path. On τ²-bench (airline / retail / telecom) with a GPT 5.4 agent + verifier: **Pass⁴ 0.42 → 0.62** mean; the largest gain on telecom (most workflow-structured): **0.19 → 0.61**. Same workflows transfer to Claude Sonnet 4.6 and Gemini 2.5 Pro agents.

**Prereqs:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md)
**Related:** [env-harness.md](env-harness.md), [../safety/safety-case.md](../safety/safety-case.md)

---

## What it is

A runtime-enforcement mechanism that treats **policy as a first-class artifact**:

- **Workflow graph.** Each domain policy is compiled into a graph of steps, edges (allowed transitions), and per-step requirements (evidence, confirmations, prerequisites).
- **Persisted state.** As the agent takes actions and gathers user information across turns, the current position in the graph — and which requirements are open, satisfied, or violated — is persisted.
- **Proactive verifier.** At each user-turn boundary, a verifier reconciles the graph state against the conversation, and either issues **step-specific remediation** (a hint that keeps the agent on the compliant path) or accepts the current step.

## How it works

1. **Compile.** Convert the natural-language policy document into a workflow graph (step nodes, transition edges, per-step requirements). Compilation is one-shot per policy.
2. **Rollout.** The agent acts as usual, consulting the graph as context.
3. **Verify at user-turn boundaries.** Between user turns, the verifier:
   - Advances graph state given the agent's actions and gathered info.
   - Detects violations (skipped required identification, taken a forbidden transition).
   - Emits **remediation** — a targeted, step-specific hint — routed back to the agent to bring the next turn onto a compliant path.
4. **Same graph, many agents.** Because the graph and verifier are external to the agent LLM, the same compiled policy transfers across model families with no retraining.

## Why it matters

Business-deployed agents run into "the model was helpful but wildly out of policy" as their dominant failure mode. Runtime guardrails historically focus on the *action* — but procedure violations (didn't ID first, didn't disclose the terms) are procedural, not action-local. Compile-once, verify-per-turn workflow enforcement is a general pattern that treats policy as an artifact you can version, audit, and reuse across agents — the natural counterpart of RLVR's verifier idea, applied to compliance rather than correctness.

## Gotchas & tricks

- Compilation quality is the whole game — a workflow graph that misses a policy branch will pass agent behavior on that branch as compliant.
- The largest gains are on the most workflow-structured domain (telecom, 0.19 → 0.61); domains with less procedural structure (airline, 0.42 → 0.62 mean) show smaller gains — the tool's leverage is proportional to how much procedure there is to enforce.
- Verifier latency at every user-turn boundary is added conversational cost; the paper does not report per-turn latency budgets.

## Sources

- Paper: *PolicyGuide: From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents* — Yu, Hwang (KAIST / DeepAuto.ai), 2026 — [arXiv:2608.19861](https://arxiv.org/abs/2608.19861)
