# Proactive Multi-Problem Discovery
*Depth — turn an LLM agent from a reactive question-answerer into a finder of problems the user didn't ask about.*

**TL;DR:** A reactive agent reads a context and answers exactly what was asked. A proactive agent reads the same context and *surfaces other problems lurking in it* — bugs, inconsistencies, hidden risks — without being prompted to. **TIDE** (Jeong et al., 2026) shows the trick is two coupled mechanisms: iterative discovery (small batches of candidates per round, each round conditioned on what's already found) and *thought templates* (reusable schemas distilled from solved cases that anchor each prediction in a recognisable problem class).

**Prereqs:** [agents/README.md](./README.md), [post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

Most LLM-agent eval suites — and most deployments — measure performance on *one explicit request*. TIDE generalises the task: given a context (a workspace, a repository), surface the **set** of problems present, where the total count is unknown, problems are evidence-grounded, and each problem is paired with a concrete action.

The framing matters: single-shot LLMs anchor on the most salient case and emit generic claims, missing the long tail. Parallel multi-agent calls produce redundancy without coverage. The right primitive is iterative discovery with explicit memory.

## How it works

### Iterative discovery

```
found = []
for round in range(N):
    candidates = LLM.surface(context, already_found=found)   # k per round
    candidates = verify(candidates, context)
    found.extend(candidates)
```

Each round emits a small batch (e.g. 3–5) of candidate problems conditioned on `found`. Conditioning on past finds steers the model away from the salient cases it would otherwise re-emit, so subsequent rounds extend coverage. The structure is the opposite of best-of-N sampling: not "pick the best one," but "find a different one."

### Thought templates

A thought template is a reusable schema mined from previously solved cases. Each template specifies:

- the *signal* in the context that triggers it (e.g. "function declared but never called");
- *how* to connect the signal to a candidate problem;
- the *class* of problem this would be.

At inference, the agent retrieves a small set of templates relevant to the current context and instantiates each. Templates anchor predictions in recognisable problem classes, preventing "generic claim" failures.

### Verification

A lightweight verifier (rule-based or LLM-judge) checks that each surfaced candidate is grounded in evidence the context actually contains. Unverified candidates are dropped before the next round so they don't poison `already_found`.

## Why it matters

- **Closer to real usage.** Users don't enumerate every problem in a workspace; they expect an assistant to notice. Proactive discovery is the closer cousin of code review, audit, and "what should I worry about" assistants.
- **Compositional with tool agents.** Each surfaced problem can be paired with an action (fix, file a ticket, alert the user) — proactive surfacing slots cleanly upstream of existing tool-calling agents.
- **A direction for agent benchmarks.** Most existing benchmarks score one-shot answers; TIDE is a worked example of evaluating coverage of an unknown problem set.

## Gotchas & tricks

- **`already_found` grows.** After many rounds the conditioning prompt is long. Either summarise it or cap rounds.
- **Verifier quality dominates.** A weak verifier lets the agent inflate `found` with hallucinations, which then suppress real problems via the conditioning signal. Strict verification first, recall second.
- **Templates aren't free.** They have to be distilled from a corpus of solved cases — without that corpus, TIDE reduces to plain iterative discovery and the gains shrink.
- **Round termination is open.** The total problem count is unknown; declaring "no more problems" requires a stopping criterion (saturation, declining novelty, fixed budget).
- **Not the same as best-of-N.** Sampling more candidates per round helps a little; *conditioning subsequent rounds on prior finds* is the structural lift.

## Sources

- Paper: *TIDE: Proactive Multi-Problem Discovery via Template-Guided Iteration* — Jeong, Baek, Kang, Hwang (KAIST), 2026 — [arXiv:2606.04743](https://arxiv.org/abs/2606.04743).
