# Agent Safety Harness (evolved)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **system-level enforcement layer** wrapping an LLM agent — policy text plus executable checker code — that filters and constrains the agent's actions. Instead of one hand-written harness applied everywhere, EvoSafeHarness *evolves* a model- and domain-specific harness per deployment, defending both indirect prompt injections and direct harmful requests without collapsing utility.

**Prereqs:** [_attacks](_attacks.md), [../agents/README](../agents/README.md)
**Related:** [safety-case](safety-case.md), [cot-monitoring](cot-monitoring.md)

---

## What it is

An agent safety harness is code that sits **outside** the LLM: it inspects proposed actions, tool arguments, and returned observations, and rejects, sanitizes, or requests confirmation. It composes with model-level defenses (RLHF for refusal, constitutional AI) — the model can still be jailbroken; the harness catches unsafe execution.

The key move here is that harnesses are usually written once by experts and shipped to every agent. Different agents (browsing / coding / customer support) have wildly different threat surfaces, so a shared harness either over-restricts everything or under-restricts most things. EvoSafeHarness treats the harness itself as an artifact to optimize — per model, per domain.

## How it works

- **Representation.** A harness is a pair `(policy: text, enforcement: code)`. The policy states what is allowed / forbidden in the domain; the enforcement is executable code that queries the LLM or applies structural checks against each proposed action.
- **Fitness function.** Score a candidate harness on a benchmark suite combining (a) attack success rate over indirect prompt-injection scenarios and direct harmful requests, and (b) utility on benign task benchmarks. Fitness = utility − α · attack-success.
- **Evolution loop.** Mutate/crossover both policy text and enforcement code (LLM-driven edits). Select the top candidates each generation. Run for K generations against the target `(model, domain)` pair.
- **Deployment.** The resulting harness is training-free from the model's perspective — dropped in front of the agent loop as an online moderator. It scales to any agent controller because it runs as a wrapper.

## Why it matters

Frontier-API moderation on every action is expensive and doesn't fit edge/on-prem deployments. Hand-authored harnesses don't scale to the diversity of agent products. Auto-evolving policy+code gives a plausible answer to "who writes all these harnesses?" — and per-deployment harnesses drop attack success materially without hurting utility, in the paper's benchmarks.

## Gotchas & tricks

- **Evolve both text and code.** Policy-only evolution under-defends; code-only evolution over-restricts. Coupling them lets the code enforce exactly what the policy says.
- **Utility must be part of fitness.** A harness that refuses everything has zero attack success and zero use. Balance is set by α; the paper finds α needs re-tuning per domain.
- **Per (model, domain) matters.** A harness evolved on GPT-class agents does not transfer well to open small-model agents — the failure modes differ.
- **Beware harness-side prompt injection.** If your enforcement code calls the same LLM to judge actions, indirect injection can attack the *judge*. Structural checks (regex, schema, sandbox) should carry as much weight as LLM-based judgments.
- **Evolution is not a one-shot.** Rerun each release cycle — new model behavior, new attacks, and new tools all shift the fitness landscape.

## Sources

- Paper: *EvoSafeHarness: Evolving Model- and Domain-Specific Harnesses for Securing Agents* — Li, Ma, Cao, Suh, Li, Song, Xiao, 2026 — https://arxiv.org/abs/2609.05903
