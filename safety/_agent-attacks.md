# Agent Attacks

*Taxonomy — attack classes specific to tool-using LLM agents, where the target is the action layer, not just the content layer.*

**TL;DR:** Agents multiply the LLM attack surface. Beyond the model's refusal behavior, an attacker can compromise the *environment* the agent perceives, the *tools* it invokes, or the *evidence chain* that authorizes its actions. This taxonomy separates injection-into-environment (indirect prompt injection), tool-layer abuse (spoofed tools, malicious schemas), memory poisoning (persistent state), and privilege abuse (misusing granted permissions). Sits under [_attacks](_attacks.md) as the "agent misuse" leaf.

**Related taxonomies:** [_attacks](_attacks.md), [_jailbreaks](_jailbreaks.md)
**Depth files covered here:** [indirect-prompt-injection](indirect-prompt-injection.md), [runtime-contract](runtime-contract.md)

---

## The problem

A jailbreak makes a model *say* something bad. An agent attack makes a system *do* something bad. Once a model has tool access — shell, browser, mail, code execution, funds — the harm ceiling jumps by orders of magnitude, and the attacker no longer needs to control the user prompt to weaponize it. The old jailbreak taxonomy under-covers this regime.

## The shared pattern

Every agent attack fits the schema: **untrusted content lands in a channel the agent trusts, and the agent turns it into an action.** The variants differ in *which* channel and *how* the action gets executed.

## Variants

| Attack class | Channel exploited | Depth file |
| --- | --- | --- |
| [indirect-prompt-injection](indirect-prompt-injection.md) | Retrieved docs, tool outputs, env state, multimodal inputs | ✅ |
| Tool spoofing / malicious schema | Tool-registry entries the agent lists (MCP servers, function schemas) | no depth file yet |
| Environment-state attacks (EMHA-style) | Long-horizon accumulated state across tool calls | no depth file yet |
| Memory poisoning | Long-lived agent memory / RAG index the agent reads back | no depth file yet |
| Privilege abuse / over-scoped permissions | The permission model itself — no injection needed, just misuse | no depth file yet |
| Runtime contract violation | Actions that lack authorization / evidence per policy | [runtime-contract](runtime-contract.md) (defense side) |

## How to choose (as a defender)

Match defenses to the exploited channel:

- **Injection into retrieved / tool content** → sanitize, quote, and treat all non-operator text as data. Prefer content-signed inputs where feasible.
- **Environment-state attacks** → invariants over agent-visible state, checkpointed rollback, human-in-the-loop for state mutations that cross privilege boundaries.
- **Tool spoofing** → cryptographically pinned tool registries, hash-verified schemas, deny-by-default for newly listed tools.
- **Memory poisoning** → memory-quarantine per session, provenance on every stored fact, per-fact-source trust scoring.
- **Privilege abuse** → least-privilege scoping at tool-declaration time, action dry-run, [runtime-contract](runtime-contract.md)-style evidence requirements before execution.
- **Any of the above** → trajectory-level [cot-monitoring](cot-monitoring.md) as the last-line-of-defense; alignment-only defenses will not close these gaps.

## Adjacent but distinct

- [_jailbreaks](_jailbreaks.md) — refusal-bypass at the model layer. Complementary but the action layer is where agent attacks live.
- [_scheming](_scheming.md) — the model itself is adversarial. Here the adversary is external; the model is a victim/proxy.
- Extraction / poisoning / backdoors ([_attacks](_attacks.md)) — target the training pipeline or model internals. Agent attacks target the *deployment*.

## Sources

- *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* — Greshake et al., 2023 — the founding paper for the injection branch.
- *OpenART: Scaling Agent Red Teaming via Open-Ended Environment Evolution* — Chen et al., 2026 — environment-state attacks at scale.
- *ToolHazard: Scaling Adversarial Environments for Security Evaluation and Alignment of LLM-based Agents* — Mou et al., 2026 — auto-generated executable environments + payloads + alignment data.
- *Agent Safety Should Be a Runtime Contract* — Ng et al., 2026 — 52-incident audit motivating runtime-contract defenses.

---

## Conventions

- **Filename:** `_agent-attacks.md` (leading underscore — taxonomy).
- **Folder placement:** `safety/`, sibling of [_attacks](_attacks.md) and [_jailbreaks](_jailbreaks.md).
- **Scope:** agent-deployment-specific attacks; extend as new depth files land.
