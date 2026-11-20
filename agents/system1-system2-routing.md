# System-1 / System-2 Routing for Agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Compile an agent workflow's human-written skill procedures into a set of **decision points**, train one small calibrated model (System-1) to handle the easy routine decisions in a single forward pass, and **escalate only uncertain cases** to a large LLM (System-2). AnyS1 (2026) is the canonical implementation: one 4B System-1 model, two System-2 families, three unseen benchmarks — 75–91% cheaper episodes and 63–92% less model-call time than an uncompiled System-2 agent, with task reliability preserved.

**Prereqs:** [README.md](README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

Modern LLM agents call a large model at every step: tool choice, argument filling, next-action decision, re-check. Many of those decisions are trivial — picking a button, formatting an argument, choosing between two obvious options. Paying System-2 LLM cost for each is wasteful.

System-1 / System-2 routing: a cheap, fast, trained small model handles the obvious decisions; when it's uncertain, the task escalates to the large LLM. Mirrors the speculative-decoding pattern — small model guess, verify / fall back to big model.

---

## How it works

1. **Compile procedures into decision points.** A human-written skill procedure (a "how to do task X" doc) becomes a graph of branching decisions. Each branch is a decision point.
2. **Train a calibrated System-1 model.** A small (e.g. 4B) LLM is SFT'd + preference-trained on correct decisions at each decision point. Critically, train with a **confidence head** so the model can abstain reliably.
3. **At runtime**, the System-1 model takes the current state + decision point and emits either a decision + confidence, or an *abstention*.
4. **Routing policy.** If confidence > threshold, use System-1's decision. Else, escalate to a System-2 LLM.
5. **Calibration control.** Confidence is tuned per deployment to hit a target escalation rate (equivalent to target cost reduction).

---

## Why it matters

- **Cost reduction at parity.** Reported 75–91% episode-cost reduction and 63–92% model-call-time reduction with preserved task reliability across unseen workflows.
- **Natural fit for the agent regime.** Most agent step budgets are dominated by a few important decisions; System-1 absorbs the rest. Analogous to how MoE routing or speculative decoding exploit per-token / per-call heterogeneity.
- **Composable with other efficiency primitives.** System-1 routing is orthogonal to speculative decoding, quantization, MoE, and KV cache tricks. Multiplicative savings in practice.

---

## Gotchas & tricks

- **Calibration is the whole thing.** An overconfident System-1 silently degrades agent reliability. Test calibration under distribution shift — procedure drift and new tool outputs regularly break it.
- **Procedure compilation is human labor.** The System-1 model is trained on a fixed decision graph. Adding new skills requires extending the graph.
- **Escalation threshold is per-cost-target, not per-quality.** Expose it as a knob at deployment.
- **Guardrails still on the big model.** Safety-critical decisions (destructive actions, financial actions) should be escalated unconditionally, not routed through System-1.
- **Not the same as "small model for easy inputs."** Routing happens per decision point in a procedure, not per user query — System-1 can handle three decisions in a workflow where System-2 handles one.

---

## Sources

- Paper: *Thinking Fast, Anywhere: One System-1 for the Decision Points of Any Agent Workflow* — Li, Zhang, Huang, Liu (USTC / NTU) — 2026 — [arXiv:2611.23751](https://arxiv.org/abs/2611.23751).
- Related pattern: speculative decoding (Leviathan et al., 2023) — small-model-first, big-model-verify.
- Related evaluation: AgentBench, τ-bench — long-horizon agent benchmarks.
