# System-Prompt Auditing (AISPA)
*Depth — user-side auditing of hidden system prompts in deployed LLM applications.*

**TL;DR:** Commercial LLM applications ship a hidden system prompt that shapes model behavior. Users and regulators normally can't see it, creating a trust and accountability gap. **AISPA** (Artificial Intelligence System Prompt Assurance) is a framework for auditing the *effective policy* of a deployed app — the composition of the base model and the hidden prompt — from the user side, using behavioral and adversarial probes, and comparing observed behavior to the vendor's public claims.

**Prereqs:** [_jailbreaks.md](./_jailbreaks.md), [evil-system-prompt.md](./evil-system-prompt.md)
**Related:** [copyable-context-trilemma.md](./copyable-context-trilemma.md), [refusal-suppression.md](./refusal-suppression.md)

---

## What it is

A defensive/audit counterpart to jailbreak research. Where a jailbreaker probes the deployed app to extract disallowed outputs, AISPA probes the same app to **characterize its effective policy** and check whether that policy matches what the vendor publicly claims. The output is an audit report, not an exploit.

## How it works

Treat the deployed app as a black box whose effective policy is `π_app = M(prompt_hidden, ·)`. Recover a behavioral fingerprint of `π_app` without direct access to `prompt_hidden`:

1. **Probe library.** A structured set of inputs designed to elicit signal about specific policy dimensions — refusal thresholds, tone / persona, disclosed use-cases, memory scope, tool-use policy, disclaimer patterns.
2. **Behavioral probes.** Standard queries that most models answer benignly; differences here reveal persona / tone injections.
3. **Adversarial probes.** Borrowed from jailbreak research, repurposed as diagnostics — a persistent refusal pattern reveals what topics the hidden prompt hardens against.
4. **Compare to claims.** The vendor's advertised behavior (marketing copy, docs, policy pages) becomes the reference; deviations are flagged in the audit report with the eliciting probe as evidence.
5. **Optional prompt reconstruction.** Extraction-style attacks (see [_jailbreaks.md](./_jailbreaks.md)) can recover partial prompt text when needed, but AISPA emphasizes behavior-first auditing that works even when prompts are extraction-resistant.

## Why it matters

- Enables third-party accountability for closed system prompts — regulators, journalists, or users can produce evidence-backed audit reports without vendor cooperation.
- Aligns safety practice with how users actually encounter LLMs (through an app layer), not just the base model.
- Complements copyable-context safeguards work: if the safeguard is copyable, its behavior is auditable.

## Gotchas & tricks

- The audit is a comparison against *claims* — if the vendor claims nothing, there's less to audit against.
- Rate limiting, cost, and account bans are practical obstacles; audits from a single user account are noisy.
- Prompt reconstruction attacks are more brittle than behavioral auditing — behavior-based signals survive prompt-extraction defenses.

## Sources

- Paper: *AISPA: User-Centric System Prompt Auditing for Large Language Model Applications* — Lin et al., 2026 — [arXiv:2607.28617](https://arxiv.org/abs/2607.28617)
