# Agent Safety Harness (Evolved NL + Code)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A hybrid safety filter for LLM agents that combines **natural-language directives** (LLM-checked) with **executable code checks** (deterministic), evolved automatically per model and per domain by an optimization loop against attack + utility benchmarks. **EvoSafeHarness** (2026) reports 10.0% attack success on DecodingTrust-Agent with utility preserved, and 82.8% utility at 0% ASR on AgentDojo — with transfer to unseen domains.

**Prereqs:** [_attacks.md](./_attacks.md), [_jailbreaks.md](./_jailbreaks.md)
**Related:** [cot-monitoring.md](./cot-monitoring.md), [../agents/README.md](../agents/README.md)

---

## What it is

Two extremes have dominated agent safety:

- **Model-side alignment** — bake refusal into the policy via RLHF/constitutional-AI training. Opaque, hard to patch per deployment, and every capability boost erodes it.
- **Handcrafted allowlists / regex filters** — deterministic and inspectable, but brittle to phrasing and impossible to keep up with novel attacks.

An **agent safety harness** is a middle path: a *composable* filter that sits in front of each agent action and combines both flavors:

- **NL directives** — short policies phrased in natural language, checked by an LLM against the proposed action. Good for context-dependent judgments ("does this action leak sensitive info?").
- **Code checks** — deterministic assertions in code. Good for stable invariants ("no shell command containing rm -rf on user data").

Because the harness is a *composable artifact*, it can be optimized end-to-end.

## How it works

EvoSafeHarness treats the harness itself as the object of optimization:

1. **Represent a harness** as an ordered list of (NL directive, code check) items with a routing rule that decides which check applies per action type.
2. **Score** each harness variant on (a) an attack benchmark (attack success rate down = good) and (b) a utility benchmark (task completion rate up = good). Both benchmarks are agent-loop simulations, not static QA.
3. **Evolutionary search** — mutate variants by inserting, editing, or reordering items (LLM-proposed edits), keep Pareto-optimal ones, and iterate.
4. **Deploy** the surviving harness as a synchronous wrapper around the agent's tool-call step.

Harnesses evolved on one agent + domain transfer to related agents (base model changes) and related domains, suggesting the evolved directives are capturing generalizable patterns rather than benchmark-specific tricks.

## Why it matters

- **Patchability.** When a new attack lands, the fix is a new harness item — no re-training the base model, no waiting for the next release cycle.
- **Interpretability.** Both the NL directives and code checks are human-readable; a security reviewer can audit a harness in an afternoon. Contrast with a black-box alignment training run.
- **Empirically strong Pareto frontier.** 82.8% utility at 0% ASR on AgentDojo is a much better tradeoff than either model-side alignment or filter-only baselines report on the same benchmark.
- **Transferable.** Evolved harnesses don't collapse when deployed on new agents; they can be a shared organizational asset.

## Gotchas & tricks

- **Evolution needs a strong attack benchmark.** If the attack set is narrow, the harness overfits — evolving toward known vectors and missing the next one. AgentDojo + DecodingTrust-Agent are the current standards; expect them to be superseded.
- **NL directives can drift into circular checks.** "Do not perform harmful actions" ⇒ the LLM checker restates the same intent to itself. Directives must be phrased operationally (naming action types, targets, contexts).
- **Code checks pay per-action latency.** A large code-check suite serialized per action becomes the bottleneck; the harness typically batches or short-circuits based on action classification.
- **Not a replacement for base-model safety training.** Harnesses handle the deployment-time layer; base-model refusal training is still the last line when the harness misclassifies an action. Both layers are needed.
- **Composable ≠ trivially composable.** Adding a new harness item can silently disable an existing one via routing rules; re-scoring after every merge is essential.

## Sources

- Paper: *EvoSafeHarness: Evolving Model- and Domain-Specific Harnesses for Securing Agents* — Li, Ma, Cao, Suh, Li, Song, Xiao (UChicago, NVIDIA, Cornell, UC Berkeley), 2026 — arXiv:2609.05903.
