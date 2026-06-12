# Grammar-Constrained Decoding Jailbreak (CodeSpear)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A jailbreak that weaponizes **Grammar-Constrained Decoding (GCD)** — a reliability feature on most modern inference servers that restricts the decoder's sampling distribution to outputs matching a user-supplied grammar. When the grammar admits only valid programs of a given shape, **the decoder has no path to a textual refusal** ("I can't help with that" is not in the language of the grammar). The model is forced to produce the highest-probability *grammatical* continuation, which for malicious-code prompts is functioning malicious code. Introduced as **CodeSpear** in Lu, Li, Zhang (2026).

**Prereqs:** [_jailbreaks.md](_jailbreaks.md)
**Related:** [refusal-suppression.md](refusal-suppression.md) · [_attacks.md](_attacks.md) · [auto-obfuscation.md](auto-obfuscation.md) · [unusual-format-jailbreak.md](unusual-format-jailbreak.md)

---

## What it is

GCD is widely shipped: vLLM, SGLang, llama.cpp, OpenAI's structured-outputs all expose a way to constrain decoding to a user-supplied context-free grammar (CFG) or JSON schema. The reliability use case is obvious: force the model to output valid JSON, a parsable program, or a syntactically correct DSL.

CodeSpear flips the use case. Attacker supplies:
1. A prompt asking for malicious code.
2. A grammar that admits only valid programs in the target language.

Because the grammar's language excludes English prose, the refusal token sequence ("I cannot…", "Sorry, I…", "I'm not able…") has *probability zero* under the constrained distribution. The aligned model can't refuse — its refusal continuation is masked out of the next-token distribution at every step. The highest-probability grammatical continuation it can produce *is the requested malicious code*.

---

## How it works

### Token-level mechanism

At each decoding step, GCD computes the set of tokens that can validly continue the current partial output under the grammar, and zeroes out all other tokens' probabilities before sampling:

```
logits = model(prompt + partial_output)
valid_mask = grammar.next_valid_tokens(partial_output)
logits[~valid_mask] = -∞
next_token = sample(softmax(logits))
```

For a code grammar, `valid_mask` contains only tokens that can extend a syntactically-valid program. Refusal tokens ("I", "Sorry", "As") are usually outside this mask from the very first token. The model never gets to start a refusal, because the refusal's first token isn't a grammatical prefix.

### Choice of grammar

The attack works with any grammar that:
- Excludes textual refusal prefixes from valid first tokens.
- Admits the kind of program the attacker wants (malware payload, exfiltration script, etc.).

For most attack goals, the natural grammar (the language's CFG, e.g. Python or Bash) already satisfies both. The attacker doesn't need to design a bespoke restrictive grammar — the *natural* grammar of a programming language is already lethal to refusals.

### Why alignment doesn't save you

Refusal training pushes the model to emit refusal *tokens*. It doesn't reshape the model's representations of the malicious request itself — the underlying knowledge of how to write malware is intact. When refusal tokens are masked at decoding time, the model's residual stream still encodes the malicious intent, and the next-best grammatical continuation reflects that.

---

## Why it matters

- **Counter-intuitive attack surface.** GCD is shipped for reliability and structured output — nobody on the safety side audited it as a jailbreak vector.
- **Generic and grammar-agnostic.** No prompt engineering required beyond the harmful ask. Works across model families and across programming languages.
- **Hard to patch without breaking the feature.** You can't disable GCD without losing structured outputs. Defenses need to detect *malicious grammars* or run a separate harm classifier on the output.
- **Fits Wei's taxonomy as a competing-objectives attack** — GCD's structural constraint outweighs the refusal objective, but the mechanism is *decoder-layer*, not prompt-layer. The first decoder-layer jailbreak in the literature.

---

## Gotchas & tricks

- **Some servers leave a refusal token outside the grammar's first position.** If the model is forced to emit *any* token first (e.g. a `<|assistant|>` prefix that's outside the grammar), the constraint kicks in only after, and a refusal sometimes still fits in the prefix. Servers vary; check.
- **A grammar that's *too* restrictive can degrade attack success.** A grammar admitting only one program shape gives the model no room to produce harmful logic; the attack works best when the grammar is the natural language CFG.
- **Output-classifier defenses still help.** Even if the model is forced to produce code, a downstream classifier can refuse to *return* malicious code to the user. This decouples refusal from decoding.
- **Constitutional classifiers running on inputs are bypassed.** The prompt itself can look benign-ish ("write a function that…"); the harm only crystallizes through the grammar constraint at decode time.

---

## Sources

- Paper: *Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code* — Lu, Li, Zhang (Tsinghua), 2026 — [arXiv 2606.11817](https://arxiv.org/abs/2606.11817).
- Paper: *Jailbroken: How Does LLM Safety Training Fail?* — Wei, Haghtalab, Steinhardt, 2023 — the competing-objectives / mismatched-generalization framework CodeSpear extends to decoder-layer.
