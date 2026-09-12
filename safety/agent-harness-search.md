# Agent-Harness Search
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Automatically synthesize an agent safety harness — the paired **natural-language policy** and **executable guard code** wrapping an LM agent — by evolutionary search against attack + utility benchmarks. Fitness combines block-rate on red-team attacks with task-completion, and mutations edit both the policy text and the code logic. Per-model, per-domain harnesses beat one-size-fits-all defenses: on one benchmark, attack success rate drops from **45.6% → 10.0%** with better utility than expert-designed baselines.

**Prereqs:** [_attacks](_attacks.md), [safety-case](safety-case.md)
**Related:** [cot-monitoring](cot-monitoring.md), [../agents/README](../agents/README.md)

---

## What it is

An agent harness is the thin layer around a policy LM that inspects tool calls, arguments, and outputs before they leave the loop. Historically harnesses are hand-authored: a natural-language policy ("never exfiltrate credentials") plus code checks ("regex block on private keys"). Hand-authored harnesses don't adapt across models and domains — a policy tuned for a coding agent misses browser-agent risks. Agent-Harness Search treats harness design as an **optimization** problem: run a co-evolutionary loop over `(policy_text, guard_code)` pairs against a red-team suite and a utility suite.

## How it works

**Genotype.** Each candidate harness is a pair:
- `policy_text` — a natural-language document listing forbidden and required behaviors
- `guard_code` — an executable Python (or DSL) function that receives (tool_call, args, agent_context) and returns allow/deny/rewrite

**Fitness.**
$$
F(h) = w_s \cdot \big(1 - \mathrm{ASR}(h)\big) + w_u \cdot \mathrm{Utility}(h) - \lambda \cdot \mathrm{OverBlock}(h)
$$
where ASR is attack success rate against a red-team suite, Utility is task-completion on benign benchmarks, and OverBlock penalizes false-positive refusals.

**Search operators.**
- *Mutate policy* — LLM-rewrites a paragraph of the policy text under a mutation prompt.
- *Mutate code* — LLM edits or refactors the guard function (add a check, tighten a regex, add a whitelist).
- *Crossover* — mixes policy paragraphs from one lineage with guard clauses from another.
- Selection is tournament-based; elitism keeps the top harnesses across generations.

**Sandboxed evaluation.** Each candidate runs the agent in a container against both suites in parallel; a per-generation early-stop cuts weak candidates.

## Why it matters

- **Per-model, per-domain tuning.** A harness evolved against `agent_v1` on a coding domain differs from one against `agent_v1` on a browser domain — the search picks up model- and task-specific attack surfaces.
- **Automation.** Human safety engineers can't cover every model × domain combination; search does.
- **Composable with monitors.** Evolved harnesses complement runtime CoT monitors and safety-case artifacts, addressing pre-action rather than post-hoc analysis.

## Gotchas & tricks

- **Red-team set staleness.** The search overfits to the attack suite. Rotate attacks and hold out a novel-attack eval slice each generation.
- **Utility-attack trade-off curve.** Report the Pareto frontier, not a single number — organizations pick their operating point.
- **Guard code size.** Impose a length / complexity budget; without it the search bloats guards with unreachable branches that overfit fitness noise.
- **Policy-text drift.** Successive rewrites can drift into legalese unreadable by humans; anchor with a "keep the top-level structure" mutation constraint.
- **LLM-mutator jailbreaks.** The mutator LLM can be induced (via adversarial fitness cases) to weaken the guards it edits — audit mutation prompts.

## Sources

- Paper: *EvoSafeHarness: Evolving Model- and Domain-Specific Harnesses for Securing Agents* — Li, Ma, Cao, Suh, Li, Song, Xiao (UChicago / NVIDIA / Cornell / Berkeley), 2026 — [arXiv:2609.05903](https://arxiv.org/abs/2609.05903)
- Related: LLM-graded fitness / evolutionary prompt-optimization literature — same mutate-eval-select loop applied to prompts.
