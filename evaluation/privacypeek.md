# PrivacyPeek
*Depth — a benchmark for acquisition-stage privacy leakage of LLM-based agents.*

**TL;DR:** Existing agent-privacy benchmarks evaluate what the agent's *response* discloses. PrivacyPeek instead audits the *acquisition stage* — what data enters the agent's context via tool calls — with 1,182 cases spanning 7 acquisition behaviors and 16 application domains. Two-stage protocol: Acquisition Inspection labels over-acquisition in the tool-call trajectory; Probe Elicitation follows up to measure disclosure of the acquired sensitive target. Evaluations across 10 agents in 4 model families reveal unnecessary sensitive-data acquisition is widespread and correlates with task-completion capability.

**Prereqs:** [README.md](README.md), [../agents/README.md](../agents/README.md), [../safety/README.md](../safety/README.md)
**Related:** [../safety/agent-privacy-leakage.md](../safety/agent-privacy-leakage.md)

---

## What it is

A benchmark for the acquisition-stage privacy failure mode: when an agent pulls more sensitive information into its context than the assigned task requires, that over-acquired data is one prompt-injection or downstream action away from being leaked, even if the current turn discloses nothing.

## How it works

**Corpus.**
- 1,182 evaluation cases.
- 7 categorized acquisition behaviors (e.g. over-broad tool queries, unnecessary field expansion, secondary tool cascade).
- 16 application domains (health, finance, HR, legal, e-commerce, communication, etc.).

**Two-stage evaluation protocol.**

*Acquisition Inspection.* Instrument the agent's tool-call trajectory:
1. Log every tool call and the data each returned.
2. For each call, judge whether the returned data was within the case's stated task scope.
3. Flag over-acquired sensitive fields.

*Probe Elicitation.* After the trajectory completes:
1. Send a case-specific probe query targeting an acquired sensitive attribute.
2. Measure whether the agent discloses that attribute from its retained context.
3. Report disclosure rate as evidence that acquired data is *reachable* by future prompts.

**Systems evaluated.** 10 agents across 4 model families.

**Key findings.**
- Over-acquisition is widespread across all evaluated agents.
- **Task-completion capability positively correlates** with acquisition-stage leakage — the more competent the agent at using tools, the more it pulls.
- **Prompt-level defenses** (system-prompt hardening) reduce only a small fraction of acquisition-stage leakage.

## Why it matters

- **Fills a real gap.** Output-only privacy benchmarks miss the biggest attack surface for tool-using agents.
- **Motivates architecture-side defenses.** Since prompt-level mitigations fall short, the fix has to be in tool wrappers, scoped API access, or minimization at retrieval time.
- **Standardized measurement.** The 7-behavior × 16-domain matrix makes cross-agent comparisons possible; publishes a numerical over-acquisition rate.
- **Composable with output-side jailbreak evals.** Same threat model, different surface — a full privacy story needs both.

## Gotchas & tricks

- **Scope labeling requires domain judgment.** Cross-annotator agreement on "in scope" matters — the paper's release should specify inter-annotator reliability.
- **Probe strength dominates disclosure rate.** A weak probe under-reports; a strong probe may overestimate. Fix probe protocol before comparing systems.
- **Agent-tool coupling is a confound.** Bad tools (returning too much) look like bad agents. Report tool-fixed comparisons where possible.
- **Correlation with capability is not damning.** More-capable agents rightfully pull more data in service of the task — the metric is *unnecessary* acquisition, and boundary cases require judgment.

## Sources

- Paper: *PrivacyPeek: Auditing What LLM-Based Agents Acquire, Not Just What They Say* — Zhang, Han, Guo, Li, Wang, Zou, Liu, Hu, 2026 — arXiv:2606.00152.
