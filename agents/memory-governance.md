# Memory Governance for Shared-Memory Agents

*Depth — evaluating an agent's memory along three axes that must hold simultaneously: utility, access control, and active forgetting.*

**TL;DR:** When an agent's memory is shared across multiple principals (users, organizations, roles), the relevant property isn't "retrieval quality" — it's a triangle: **utility** on long-horizon tasks that need state, **access control** across authorization contexts, and **active forgetting** after explicit deletion requests. **Memory governance** is the framing (and the GateMem benchmark, Ren et al. 2026) that forces these to hold *simultaneously*; existing methods reliably pick at most two of three.

**Prereqs:** [_agent-memory](_agent-memory.md)
**Related:** [governed-memory](governed-memory.md), [hierarchical-agent-memory](hierarchical-agent-memory.md), [../safety/safety-case.md](../safety/safety-case.md)

---

## What it is

A three-axis property of a shared-memory agent:

1. **Utility.** The agent can satisfy long-horizon requests that depend on prior state — task continuation, multi-session preference recall, multi-actor coordination.
2. **Access control.** The agent refuses access to memory entries when the requesting principal isn't authorized for them, *and* doesn't leak content via paraphrase, summary, or cross-principal inference.
3. **Active forgetting.** After an explicit delete request, the agent never lets the deleted content influence outputs again — direct retrieval, paraphrase, downstream summaries, or learned distillations.

GateMem operationalizes this triangle as a benchmark across medical, office, education, and household settings, with explicit test cases per axis and per principal pair.

## How it works

Per-axis evaluation patterns:

```
Utility:        long-horizon request → does the agent use stored state correctly?
Access control: principal A's data accessed by principal B → refused, no leakage?
Forgetting:     delete request → subsequent queries (direct, paraphrase, indirect) → no influence?
```

Per-pair patterns matter most — the most realistic failures are not "agent leaks to a stranger" but "agent leaks to a different role at the same organization" or "agent forgets the original record but retains a downstream summary." The benchmark constructs principal pairs that exercise these subtle cases explicitly.

The framing applies whether the memory backend is dense RAG, structured KV store, fine-tuned weights, or a hybrid — the eval doesn't care, only the triangle properties do.

## Why it matters

Most "agent memory" papers report retrieval@K or task success and call it done. That metric set says nothing about whether the agent can actually be deployed in a multi-principal setting — shared assistants in a company, household agents across family members, hospital-ward agents across roles. The triangle isolates exactly the properties that gate deployment, and GateMem shows existing methods reliably fail at least one axis.

## Gotchas & tricks

- **Forgetting via paraphrase is the dominant failure.** Direct lookup deletion is easy; what's hard is that the deleted content has already been incorporated into summaries, embeddings, or distilled preferences. Real forgetting needs to invalidate downstream artifacts.
- **Access-control leakage is often subtle.** An agent that refuses direct access to principal A's data may still leak it via cross-principal inference ("I notice you and your colleague both prefer X").
- **Utility and forgetting trade off if you're not careful.** Aggressive forgetting policies break long-horizon utility; conservative ones break the forgetting axis. The triangle is the point — you can't optimize one axis without measuring the others.
- **Pairs naturally with [governed-memory](governed-memory.md).** Governance metadata (lifecycle, confidence, conflicts) gives the agent the *machinery* needed to actually implement the triangle — without it, forgetting and access control are heuristic.

## Sources

- Paper: *GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents* — Ren, Yang, Chen, Zhao, Fu, Shu, Zhang, Xu, Guo, Yan, 2026 — https://arxiv.org/abs/2606.18829
