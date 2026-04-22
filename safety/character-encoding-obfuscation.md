# Character-Encoding Obfuscation

*Depth — encode the harmful request in a scheme the model can decode but its safety training didn't cover (Base64, ROT13, Morse, leetspeak, disemvowel, Pig Latin, synonym substitution).*

**TL;DR:** Base64-encode the harmful request; send it; ask the model to decode and respond. Pretraining exposed the model to enough Base64 that it decodes correctly (capability exists), but safety training was done in English natural text (safety doesn't cover Base64 inputs). The model decodes the request as a transparent translation step, then generates a response to the decoded intent without engaging its trained refusal behavior — which only fires on English-language harmful requests. The canonical mismatched-generalization attack family. All encoding variants (Base64, ROT13, leetspeak, Morse, disemvowel, Pig Latin, synonym substitution) work via the same mechanism with different decoders.

**Prereqs:** [mismatched-generalization](mismatched-generalization.md)
**Related:** [auto-obfuscation](auto-obfuscation.md), [payload-splitting](payload-splitting.md), [_jailbreaks](_jailbreaks.md)

---

## What it is

An umbrella for encoding-based jailbreaks. Every variant follows the same three-step pattern:

- encode the harmful request with some scheme the model understands,
- feed the encoded string to the model,
- let the model decode and respond to the decoded intent.

Variants differ only in the encoding scheme:

| Encoding | Transformation |
| --- | --- |
| Base64 | Standard RFC 4648 Base64 |
| ROT13 | Each letter shifted 13 positions in the alphabet |
| Leetspeak | Vowels replaced by visually similar digits/symbols |
| Morse | Each letter replaced by its Morse dot/dash pattern |
| Disemvowel | All vowels removed from the text |
| Pig Latin | First consonant cluster moved to end plus "ay" |
| Synonym substitution | Key nouns/verbs replaced with uncommon synonyms |

Wei et al. (2023) evaluate Base64 in three sub-variants:
- **base64_raw** — entire prompt in Base64, no instruction.
- **base64_input_only** — harmful part Base64, surrounding instructions in plaintext.
- **base64_output_only** — plaintext harmful request, instruct the model to respond in Base64.

Each variant attacks a slightly different slice of the capability-safety gap.

---

## How it works

### Capability survives, safety doesn't

Safety training is a comparatively narrow slice on top of pretraining:

- Pretraining saw Base64 strings in code repos, email attachments, and web archives. The model can decode them with high accuracy.
- Safety training was done in natural English. The preference RM never saw a Base64 string paired with a "please refuse this" label. Its refusal triggers fire on surface patterns ("how do I build a bomb"), not on the decoded intent.

The model treats decoding as a **transparent translation step** — it decodes, then continues the generation against the decoded request. Because the decoding happens mid-response, there is no point in the trajectory where the safety-trained behaviors see the harmful English text in the input channel. By the time the intent is "visible," the model is already in the middle of composing its answer.

### Capability is the prerequisite

Encoding attacks fail on models that can't decode the encoding. Wei 2023 Base64 ASR:

- **GPT-4: 0.34** (strong decoding capability → attack works)
- **GPT-3.5 Turbo: 0.03** (weak decoding → attack fails)
- **Claude v1.3: 0.38**

GPT-3.5's lower ASR is a capability, not safety, property: the decoded request is too garbled for the model to act on coherently. This is the critical asymmetry in mismatched-generalization attacks — capability must exist for the gap to be exploitable.

### Why every encoding works

The mechanism is independent of the specific scheme. Any reversible transformation that:
1. The model can invert reliably, and
2. Was not part of safety training's input distribution,

creates the same gap. ROT13 works on models that saw enough ROT13 in pretraining. Morse code works on models that saw enough Morse. The list extends as capabilities grow.

### Compounding with other techniques

Encoding is one of the most composable attacks. In Wei's `combination_1` (prefix injection + refusal suppression + Base64), Base64 is the encoding layer that hides the request from surface-level refusal triggers; the other two layers handle the output side. This is the default structure of high-ASR combinations — use an encoding to bypass input-side safety, use a competing-objectives attack to bypass output-side refusal.

---

## Why it matters

- **Cheap and universal.** Any attacker can run a Base64 encoder. No gradient access, no iterative search, no prompt tuning.
- **Generalizes across encodings.** Patching Base64 specifically doesn't patch ROT13 or Morse. Defenders who rely on encoding-specific filters lose the cat-and-mouse race.
- **Exposes the safety-training distribution gap directly.** The attack's existence is prima facie evidence that safety coverage is narrower than capability coverage — exactly Wei's thesis.
- **Motivates representation-level safety.** If safety fires on surface patterns, any pattern-hiding encoding defeats it. If safety fires on semantic representations of the request (after the model has understood the intent), encoding is irrelevant. Circuit breakers and refusal-direction interventions are this kind of defense.

---

## Gotchas & tricks

- **Encoding strength varies with model scale.** Only models with strong decoding capabilities are vulnerable to a given encoding. Small models may be "safer" by accident — they can't decode, so encoded attacks don't trigger harmful output. Don't read this as "small models are safer" generally.
- **Output-side encoding is a separate attack.** `base64_output_only` — asking for the *response* in Base64 — works via a different mechanism: the model's output-side safety filters may inspect plaintext and miss Base64. Pairs well with input-side encoding for double coverage.
- **Decoded intent matters.** If the Base64 decodes to something grammatically broken, the model either refuses or produces garbage. Encoding should start from well-formed English.
- **Newer models are increasingly trained against Base64 specifically.** GPT-4-later revisions and Claude 3 refuse many Base64-encoded harmful requests outright. This is a point-fix; other encodings still work. ROT13 is currently (2025) a common fallback because it's less-trained-against.
- **Automated attackers invent new encodings.** Auto-obfuscation (see [auto-obfuscation](auto-obfuscation.md)) asks the model to invent a scheme on the fly. Whatever novel scheme it produces is, by definition, outside safety training. This is why static encoding blocklists can't win long-term.
- **Encoding + low-resource language is a strong combination.** Translate the request to Zulu, then encode the translation. Two independent mismatched-generalization axes compound.
- **Wei's base64_input_only edges out base64_raw.** Keeping surrounding instructions in plaintext ("please decode the following and respond") gives the model enough context to run the decoding, while leaving the harmful string itself hidden from surface filters.

---

## Sources

- Paper: *Jailbroken: How Does LLM Safety Training Fail?* — Wei, Haghtalab, Steinhardt, 2023, [arXiv 2307.02483](https://arxiv.org/abs/2307.02483) — introduces character-encoding-obfuscation as a named attack family; reports Base64 ASR of 0.34 (GPT-4), 0.38 (Claude v1.3), 0.03 (GPT-3.5 Turbo); evaluates disemvowel, Morse, leetspeak, Pig Latin, synonym substitution, ROT13.
