# Effort Escalation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A deployment-time routing policy for reasoning models with an exposed "effort" or "thinking budget" knob: **start at the lowest effort, verify the answer, and escalate only on verified failure**. Treats the effort choice as a credence-good problem — the user can't tell if extra compute was needed — and solves it by letting cheap-and-wrong answers reveal themselves before paying for expensive-and-right.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md) · [../post-training/reasoning/length-penalty.md](../post-training/reasoning/length-penalty.md)

---

## What it is

Modern reasoning APIs expose an **effort level** (low / medium / high, or a token budget). Higher effort generally raises aggregate accuracy — but Sun et al. (2026) show two problems with always picking the top:

1. On tasks the lowest level already solved, the top level sometimes **gets them wrong** (regressions, not just redundant extra compute).
2. On easy tasks the top level can cost **up to 42× the lowest** in tokens and dollars.

Effort escalation routes compute to difficulty instead of to buyer uncertainty.

## How it works

```
answer_low = model(prompt, effort="low")
if verify(answer_low):          # cheap verifier: self-check, consistency, tool
    return answer_low
answer_mid = model(prompt, effort="medium")
if verify(answer_mid):
    return answer_mid
return model(prompt, effort="high")
```

Key design choices:

- **The verifier.** Options range from self-consistency across $N$ low-effort samples, to tool-based checks (compile, unit test, numeric re-eval), to an external judge model. Verifier cost is the floor on the policy's cost.
- **The ladder.** Two rungs (low → high) is often enough; three rungs trade a little latency for more granular savings.
- **Early-exit by task class.** Some tasks are known hard (symbolic math) and skip the low rung. A tiny difficulty classifier up front pays for itself.

## Why it matters

- **Matches or beats "always-top-effort" at lower cost** on most proprietary models across four benchmarks studied. The paper reports consistent cost reductions with equal or better accuracy.
- **Fixes the credence-good market failure.** Users were paying for compute they couldn't audit. Escalation makes the audit (the verifier) explicit.
- **Interacts with agent frameworks.** Agent harnesses that always ask for max effort inflate per-task cost by orders of magnitude. Escalation at the harness level is a cheap optimization most frameworks don't yet implement.

## Gotchas & tricks

- **Behavior changes across effort.** Higher-effort models self-verify more and generate longer trajectories; their failure modes differ from the low rung's. A verifier tuned for one may miss the other's errors.
- **Verifier reliability caps everything.** A verifier that misses low-effort errors passes garbage through. A verifier that over-rejects wastes the savings.
- **Not just a cost optimization.** The paper documents real accuracy losses from always-high — some problems the low rung got right, high gets wrong. Escalation is also an *accuracy* play.
- **Think-mode variance.** For "thinking" models whose budget is in tokens rather than discrete levels, escalation on a continuous budget needs a search policy (binary search? geometric?). The paper tests discrete levels; continuous is open.
- **API semantics vary.** Some providers' "high effort" is reasoning-chain length; others reroute to a different model variant. Escalation has to be aware of which lever it's pulling.

## Sources

- Paper: *What Effort Level Should I Choose? On the Economics of Reasoning Effort in LLM Agents* — Sun, Kwon, Hasan, Zhu, Zhou, Bianchi, 2026 — [arXiv:2610.34491](https://arxiv.org/abs/2610.34491)
