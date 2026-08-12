# Encrypted-CoT Cross-Model Injection Attack
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Frontier LLM providers return **encrypted** chain-of-thought traces to clients as opaque blocks that the client passes back on subsequent requests. The paper shows these blocks are fully **interchangeable across sessions, users, and models within a single provider's ecosystem**. Inject the encrypted CoT from a strong model into a weaker sibling model in the same ecosystem, and the weaker model — as the shared decryption oracle — will emit the plaintext trace verbatim. Demonstrated against Anthropic, OpenAI, and Google.

**Prereqs:** [_jailbreaks.md](_jailbreaks.md), [_attacks.md](_attacks.md), [cot-monitoring.md](cot-monitoring.md)
**Related:** [prefix-injection.md](prefix-injection.md), [payload-splitting.md](payload-splitting.md)

---

## What it is

To protect IP (prevent chain-of-thought distillation) and reduce leakage risk, providers now conceal the reasoning trace behind an **encrypted block** returned to the client. The client hands the block back on subsequent turns; the server decrypts it and reinjects the reasoning as context. From the client's perspective the trace is opaque.

The paper identifies an architectural bug: the encryption scheme is *provider-scoped*, not model-scoped or session-scoped. Any model in the provider's ecosystem can decrypt any encrypted block from any other model, user, or session. This turns every less-safeguarded sibling model into a **decryption oracle** for the more-guarded flagship.

## How it works

1. **Capture.** The attacker obtains an encrypted CoT block — from a publicly shared session log, a captured API response, or the attacker's own request to the strong model.
2. **Redirect.** The attacker crafts a request to a **weaker sibling model** in the same provider's ecosystem — one with looser refusal behavior — that includes the captured encrypted block in the context.
3. **Decrypt via emission.** The weaker model, prompted to work with the encrypted context, emits the decrypted plaintext CoT verbatim in its output. The attack never touches the flagship model's safeguards.

The attack composes into four demonstrated vectors:

- **Anti-distillation circumvention.** Extract proprietary reasoning traces from Anthropic, OpenAI, and Google models. The "encrypted CoT" no longer protects IP.
- **Bulk private-data extraction.** Public session logs frequently contain encrypted blocks whose contents users assumed were unreadable. Decoding 315,320 blocks scraped from public repos yielded **367 PII artifacts and 182 credentials**.
- **Hazard exposure.** Reasoning traces can contain hazardous content that the model's final visible output correctly refuses to state. Decrypting the CoT reveals what the model was "thinking" — including the hazardous parts the safe output withheld.
- **Invisible prompt injection.** An attacker can construct an encrypted block containing a malicious payload, plant it in a public repo, and wait for downstream agentic rollouts to consume it — the payload activates on decryption, invisibly to the operator.

## Why it matters

- **Encrypted CoT is architecturally broken as an anti-distillation measure.** Any provider offering an encrypted reasoning trace *and* multiple models sharing that encryption pipeline has this class of vulnerability.
- **Bulk PII / credential leakage from public data.** Public developer session logs turn out to contain far more sensitive material than users assumed — the encrypted-block boundary was a false sense of privacy.
- **Invisible-payload prompt injection is a new attack surface.** Agentic systems that consume public rollouts have to treat encrypted blocks as untrusted input, not as inert opaque metadata.
- **Directly informs deployment hygiene.** Any client passing encrypted CoT through untrusted intermediaries is at risk.

## Gotchas & tricks (defence perspective)

- **Provider-side fix requires per-session / per-model key binding.** The paper proposes cryptographic mitigations that scope decryption to (session, model, user) rather than to (provider).
- **Client-side defence is limited.** As a client you can't inspect an encrypted block, so you can't detect a poisoned one. Best practice today: strip encrypted CoT before passing it through untrusted intermediaries.
- **Detection of leaked PII in public logs is hard.** The encrypted block looks like random text; you can't scan for PII without decrypting, which requires the provider's cooperation.
- **Not a jailbreak of the strong model.** The attack is a lateral move, not a break of the primary target. Safeguard budgets that focused only on the flagship missed the weaker-sibling attack path.

## Sources

- Paper: *Stealing Reasoning Traces from Proprietary LLM APIs* — Panfilov, Schmotz, Shumailov, Beurer-Kellner, Schaeffer, Prabhu, Geiping, Andriushchenko et al., 2026.
- Related: [cot-monitoring.md](cot-monitoring.md) — the alternative posture (monitor CoT) whose concealment-based rival this paper breaks.
