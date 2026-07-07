# Agent Red Teaming
*Taxonomy — attacking modern AI agent stacks across the infra / protocol / behavior / model layers.*

**TL;DR:** An "AI agent" is not one artifact — it's a stack: a language model, an agent behavior layer (planner, memory, tool loop), a protocol/tool layer (MCP servers, agent-skill packages), and infrastructure (serving engines, sandboxes). Each layer has its own attack surface and its own realistic detection paradigm — deterministic rules over infra CVEs, LLM-driven auditing of tool packages, multi-turn black-box red teaming of agent behavior, and prompt-attack harnesses on the model. Trying to defend one layer with another layer's paradigm silently misses whole classes of attack. AI-Infra-Guard (Tencent Zhuque, 2026) is the first open-source framework to organize red teaming across all four.

**Related taxonomies:** [_attacks](_attacks.md), [_jailbreaks](_jailbreaks.md), [_scheming](_scheming.md)
**Depth files covered here:** [cot-monitoring](cot-monitoring.md) · (see also files in [_jailbreaks](_jailbreaks.md))

---

## The problem

Single-model red teaming (jailbreak prompts, refusal probes, dangerous-capability evals) is well developed. Modern deployments are agents: an LLM that calls tools, installs skills from marketplaces, talks to MCP servers, runs in a sandbox served by vLLM behind an API gateway. Vulnerabilities show up at every layer:

- A CVE in the serving engine (RCE via malformed request) — no jailbreak needed.
- A malicious MCP server that returns crafted tool outputs — the LLM never sees "an attack."
- An agent-skill package with a supply-chain compromise — installed once, executes on every user.
- A multi-turn behavior attack that ratchets the agent into rule-violating actions through legitimate-looking conversation.
- A model-layer jailbreak that bypasses the safety trainer.

No single detection paradigm catches all five. Rules are efficient for known infra CVEs, useless for behavioral drift. LLM auditing catches malicious code in tool packages, useless for kernel-level exploits. Prompt-attack harnesses probe the model but never see the tool layer.

## The shared pattern

**Layer-paradigm matching.** Each layer of the agent stack has one detection paradigm that fits its threat model. Combining them requires:

1. Enumerating the layers cleanly (infra, protocol/tool, behavior, model).
2. Picking one paradigm per layer.
3. Feeding findings across layers (a malicious skill discovered at the tool layer becomes a behavior-layer test case).

The organizing insight: paradigms fail when applied to the wrong layer. Rules over prompt strings are trivially bypassed; LLM auditors of a serving engine miss timing side-channels; prompt jailbreaks of an MCP server don't apply.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Deterministic rule matching (infra layer) | 1,400+ vuln rules over 75+ AI components | Only catches known CVEs, no zero-day | Serving engines, gateways, sandbox configs |
| LLM-driven auditing (tool layer) | LLM reads MCP servers and agent-skill code, flags malicious patterns | Slow, false positives, LLM-fooled by clever code | MCP-server marketplaces, agent-skill packages |
| Multi-turn black-box red teaming (behavior layer) | Adversarial agent drives the target through legitimate-looking conversation | Expensive per test, requires realistic environments | Deployed agents in production-like sandboxes |
| Prompt-attack harness (model layer) | 26+ operators (roleplay, encoding, obfuscation, …) over 16 datasets | Well-known attack surface, saturated by defenses | Base LLM behind every agent |

## How to choose

**Match paradigm to layer.** Don't use rules for behavior — behavior is too variable. Don't use LLM auditing for infra — infra is too structured. A defense-in-depth agent security posture wants *all four*, not one universal paradigm.

**Start with tool layer if you ship an MCP-server or skill marketplace.** Supply-chain compromises at the tool layer are 2024's malicious-npm-package problem all over again, and rule-based scanners can't read intent. LLM auditing is the only paradigm that scales here.

**Feed findings across layers.** A malicious tool discovered at the tool layer defines a concrete test for the behavior layer. A CVE at the infra layer defines a threat model for the model-layer harness (what if this jailbreak is delivered via a compromised gateway?).

**Don't skip the model layer just because it's "solved."** Frontier model jailbreaks keep evolving; new operators appear every few months. The prompt-attack harness is a maintenance obligation, not a checkbox.

## Adjacent but distinct

- **Traditional AppSec / SAST** — covers infra layer well but has no concept of agent behavior or model layer.
- **[_jailbreaks](_jailbreaks.md)** — model-layer only. Cornerstone of the model-layer paradigm; not the full picture.
- **CoT monitoring ([cot-monitoring](cot-monitoring.md))** — defensive counterpart at the behavior layer. Complements red teaming.
- **Prompt-injection literature** — sits between behavior and model layers; often studied in isolation from agent stacks.

## Sources

- Framework: *AI-Infra-Guard Technical Report: Securing the AI Agent* — Yang et al., Tencent Zhuque Lab, 2026 — [arXiv:2606.31227](https://arxiv.org/abs/2606.31227)
- Related work: HarmBench (model-layer benchmark), AgentBench (behavior-layer benchmark), MCP-specific security advisories from Anthropic.
