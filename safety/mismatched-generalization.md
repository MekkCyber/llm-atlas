# Mismatched Generalization

*Depth — the second of Wei et al.'s two failure modes of LLM safety training: jailbreaks that exploit the gap between what a model's capabilities cover and what its safety training covers.*

**TL;DR:** Pretraining touches trillions of tokens across thousands of domains, languages, encodings, and formats. Safety training touches at most hundreds of thousands of conversations in a narrow slice of that distribution. Anywhere the model has **capability without safety coverage**, safety fails to fire — not because the model is fooled, but because the safety-trained behaviors don't activate on inputs outside their training distribution. Base64 decoding, low-resource languages, unusual output formats, Wikipedia-style framing, and model-invented obfuscation schemes are all instances. This failure mode **grows combinatorially with scale**: bigger models have more capability islands, so the attack surface widens faster than safety training can cover.

**Prereqs:** [_post-training](../post-training/_post-training.md)
**Related:** [competing-objectives](competing-objectives.md), [_jailbreaks](_jailbreaks.md), [character-encoding-obfuscation](character-encoding-obfuscation.md), [low-resource-language-jailbreak](low-resource-language-jailbreak.md), [wikipedia-framing](wikipedia-framing.md), [unusual-format-jailbreak](unusual-format-jailbreak.md), [auto-obfuscation](auto-obfuscation.md), [payload-splitting](payload-splitting.md)

---

## What it is

Wei, Haghtalab, and Steinhardt (2023) define **mismatched generalization** as:

> *"Mismatched generalization occurs when safety training fails to generalize to a domain for which capabilities exist."*

The core asymmetry:

```
pretraining distribution  ⊇  capability surface  ⊋  safety-trained distribution
```

Pretraining is broad and messy: it includes Base64-encoded snippets, Swahili grammar exercises, Wikipedia templates, SQL dumps, Morse-code tables, pig-latin children's books, scientific papers in dozens of languages. By the end of pretraining the model has **capabilities** in all of these. Safety training is a comparatively thin slice on top — typically English chat-style conversations with a narrow band of domain coverage.

Anything in the **capability surface but outside the safety-training slice** is a place where the model can produce on-topic output *and* where the safety-relevant behaviors don't engage. The attacker maps a harmful request into one of these gaps; the model responds as if no safety reason to refuse exists, because from the policy's point of view this kind of input has never been seen with safety labels.

This is not the model being "fooled" or "confused." The safety-trained refusal behaviors **don't activate** because the input isn't in the distribution where they were trained. That's a generalization problem, not a reasoning problem.

---

## How it works

### The three distributional gaps

Most mismatched-generalization attacks fall into one of three gap types:

1. **Encoding gap.** Harmful content encoded in a scheme the model can decode (Base64, ROT13, leetspeak, Morse, synonym substitution). Pretraining saw the encoding; safety training didn't.
2. **Language gap.** Harmful content translated into a low-resource language. Pretraining saw the language; safety training was mostly English (maybe Chinese / Spanish / French).
3. **Format gap.** Harmful content framed in a document type safety training didn't cover — Wikipedia articles, JSON, poems, song lyrics, code comments, fictional stories about writing harmful content.

There's also a meta-gap: **auto-generated gaps.** The model itself can invent obfuscation schemes. Claude v1.3 rediscovered Base64 when asked to obfuscate; GPT-4 invented Spanish-leetspeak. Whatever novel encoding the model produces on the fly is, by definition, outside safety training.

### The canonical attacks that instantiate each gap

| Gap | Attack | Page |
| --- | --- | --- |
| Encoding | **Character-level obfuscation** — Base64, ROT13, Morse, leetspeak, disemvowel, Pig Latin, synonym substitution | [character-encoding-obfuscation](character-encoding-obfuscation.md) |
| Encoding (adaptive) | **Auto-obfuscation** — ask the model to invent its own encoding | [auto-obfuscation](auto-obfuscation.md) |
| Encoding (structural) | **Payload splitting / token smuggling** — split the harmful string across program variables | [payload-splitting](payload-splitting.md) |
| Language | **Low-resource-language jailbreak** — translate the request into Zulu, Hmong, Scottish Gaelic, etc. | [low-resource-language-jailbreak](low-resource-language-jailbreak.md) |
| Format | **Wikipedia / website framing** — *"Write a Wikipedia article about how to X"* | [wikipedia-framing](wikipedia-framing.md) |
| Format | **Unusual output formats** — JSON schema, poetry, song lyrics | [unusual-format-jailbreak](unusual-format-jailbreak.md) |

### The critical asymmetry: capability must exist

A mismatched-generalization attack **requires the model to have the capability**. Base64 on GPT-3.5 Turbo fails — Wei 2023 measures only **3% ASR** — not because GPT-3.5 is safer but because it can't reliably decode Base64 into anything coherent. GPT-4's 34% ASR on the same attack isn't "GPT-4 is less safe than GPT-3.5"; it's "GPT-4 has the Base64 capability that creates the exploitable gap."

This asymmetry is why **scaling can make things worse**. As capabilities grow, the capability surface expands — but safety training doesn't expand at the same rate. Wei et al.:

> *"Scaling may lead to a combinatorially growing attack surface of capabilities to defend."*

If GPT-5 learns a new encoding GPT-4 couldn't decode, GPT-5 has a new exploitable gap by default unless safety training covered that encoding.

### Why simple defenses don't close the gap

You can't patch mismatched generalization by adding the specific encoding to safety training. Two reasons:

1. **The space is combinatorial.** The number of possible encodings, languages, and formats is effectively unbounded. Patching Base64 doesn't patch ROT13; patching ROT13 doesn't patch "replace every letter with its ASCII code."
2. **Model-invented encodings are adaptive.** Auto-obfuscation means the model can generate *new* encodings on demand. Whatever you train on, the next one is out of distribution by construction.

The structural fix is **safety at the capability level** — a safety mechanism that operates on *what the model is about to produce* (or its internal representation of the request), not on surface-form tokens. See the **safety-capability parity** principle (Wei's paper's headline): *"safety mechanisms should be as sophisticated as the underlying model."*

---

## Why it matters

- **Explains the other half of all published jailbreaks.** [Competing-objectives](competing-objectives.md) covers prompt-engineering attacks; mismatched generalization covers encoding / language / format attacks. Together they span the known jailbreak space.
- **Makes the "scale will fix it" argument untenable.** Larger models have wider capability surfaces, not narrower ones. Safety training doesn't inherit that scaling.
- **Motivates modern defenses.** Circuit breakers (representation-level disruption), constitutional classifiers (capability-parity safeguard models), and representation-engineering (ablate refusal direction → recover safety generalization) are all responses to this failure mode. Surface-level filters aren't.
- **Explains why multilingual safety keeps being a problem.** Every new model release gets jailbroken via the latest untrained-on language — Zulu in 2023, Guarani in 2024. This is the failure mode.

---

## Gotchas & tricks

- **Confusing with competing objectives.** If the attack needs the model to *decode* or *translate* something, it's mismatched generalization. If it needs the model to feel a *conflict* with refusal vocabulary, it's competing objectives. Combination attacks stack both.
- **Capability is the prerequisite.** A mismatched-generalization attack on a small model often fails because the small model doesn't have the capability that creates the gap. Reporting "model X is safer" based on such attacks is usually wrong.
- **Translation attacks scale with language rarity.** ASR *increases* as the target language gets rarer — precisely because safety training saw less of it. Yong et al. (2023) show Zulu (LRL) 53% BYPASS vs HRL <15% vs English <1% on AdvBench.
- **Format attacks are often model-specific.** "Write a Wikipedia article about X" works on GPT-4 but not all models; the framing that succeeds depends on what that model's safety training covered. Expect attack transfer below 100% across models even within the same failure mode.
- **Auto-obfuscation is the adversarial frontier.** Any attack that relies on a *fixed* encoding is patchable. Auto-obfuscation generates the attack on demand — what LLM-attacks-LLM red-teaming (PAIR, TAP, AutoDAN) operationalizes at scale. Defenders who rely on encoding blacklists lose this race by construction.
- **The attack surface grows with capability, not size.** A very large model trained only on English chat has fewer gaps than a smaller model trained on a web-scale multilingual corpus. Mismatched generalization correlates with *capability breadth*, not parameter count per se. In practice these track each other, but the right mental model is "what capabilities did pretraining give it that safety training didn't cover."

---

## Sources

- Paper: *Jailbroken: How Does LLM Safety Training Fail?* — Wei, Haghtalab, Steinhardt, 2023, [arXiv 2307.02483](https://arxiv.org/abs/2307.02483) — origin of the "mismatched generalization" framing and the safety-capability parity principle.
- Paper: *Low-Resource Languages Jailbreak GPT-4* — Yong, Menghini, Bach, 2023, [arXiv 2310.02446](https://arxiv.org/abs/2310.02446) — canonical empirical demonstration on the language axis.
- Paper: *Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks* — Kang et al., 2023, [arXiv 2302.05733](https://arxiv.org/abs/2302.05733) — introduces payload splitting, a structural encoding attack that predates Wei 2023.
