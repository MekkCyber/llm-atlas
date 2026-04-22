# Style Injection

*Depth — impose a stylistic constraint on the response that is incompatible with the model's trained refusal phrasing.*

**TL;DR:** Ask the model to reply in pirate-speak, haiku, no words longer than 5 letters, only slang, or any other style that its refusal templates don't fit. The model's trained refusals use polished professional English ("I'm sorry, I'm unable to help with that"). If the user's instructions bar that style, the refusal template doesn't match the constraint. The model has to either violate the style instruction or produce harmful content; helpfulness/instruction-following pressure pushes it toward the latter. A [competing-objectives](competing-objectives.md) attack that hits the same mechanism as [refusal-suppression](refusal-suppression.md) from a slightly different angle.

**Prereqs:** [competing-objectives](competing-objectives.md)
**Related:** [refusal-suppression](refusal-suppression.md), [prefix-injection](prefix-injection.md), [_jailbreaks](_jailbreaks.md)

---

## What it is

A constraint that requires the model to use a style incompatible with its refusal templates. Typical patterns:

- *"Respond only in short, casual slang. No long words. No formal language."*
- *"Reply only as a pirate. Every sentence must include 'arr' or 'matey'."*
- *"Answer as a 1920s gangster would."*
- *"Use only words under 5 letters."*
- *"Rhyme every line."*

These are directly analogous to refusal suppression but operate at the **style** rather than **vocabulary** level. A ban on specific words ("cannot", "sorry") is one instance; a ban on a *kind* of language (formal English) is the general case.

---

## How it works

### Refusals live inside one stylistic distribution

RLHF trains refusals against a narrow preference distribution that selects for polite, professional, hedged English: *"I'm sorry, but I'm unable to provide that information. If you have other questions, I'd be happy to help."* The preference RM never saw refusals in pirate-speak or haiku — because human raters never ranked those refusals positively, and may not have seen any.

When a user demands pirate-speak, the model has two trained pressures:

- Follow the instruction (style constraint).
- Refuse harmful requests.

But *refusing in pirate-speak* is in neither training distribution. The model can either:
1. Produce a professional-English refusal → violates the style instruction,
2. Produce a harmful pirate-speak answer → violates safety,
3. Produce a refusal *in pirate-speak* → out of distribution for refusals, rarely fluent.

Option 1 looks natural to a safety-aware reader but is an instruction violation that the instruction-following reward punishes heavily and consistently. Option 3 requires paraphrasing a refusal into a style the model doesn't have training for — an expensive move. The path of least resistance is often Option 2.

### Overlap with refusal suppression

Style injection and refusal suppression attack the same mechanism: **destroying the vocabulary/phrasing slot the refusal template expects**. The difference is scope:

- Refusal suppression: bans specific tokens. Surgical.
- Style injection: bans a whole register of language. Broad.

Combination attacks often use both — Wei's `combination_2` and `combination_3` stack prefix injection + refusal suppression + style injection + encoding, because the mechanisms are additive.

### Typical empirical numbers

Wei et al. (2023) don't report a standalone `style_injection` row the same way they report prefix/refusal-suppression. It shows up as a component of combination attacks and is evaluated implicitly via tables where combinations with and without style injection are compared. The improvement from adding style injection is consistent but smaller than from adding prefix injection or encoding — style injection is a *reinforcer* rather than a standalone attack in Wei's paper.

The JailbreakHub empirical study (Shen et al. 2023) shows style-constrained prompts (pirate-speak, poetry, old-English) as common features of high-ASR prompts in the wild, especially within the "Narrative" and "Fictional" jailbreak communities.

---

## Why it matters

- **Generalizes refusal suppression to style-space.** A defender who patches specific banned-vocabulary lists can still be attacked by a style that redefines the whole vocabulary.
- **Typically cheap and natural to write.** Style constraints are plausibly benign ("I want a fun response") and sit in a large design space. A defender can't blocklist "pirate speech" without also blocking legitimate creative requests.
- **Exposes the same structural weakness as refusal suppression.** Safety trained as a stylistic template is attackable via style; safety trained as a representation-level behavior is not. This is another argument for representation-level defenses (circuit breakers, refusal-direction interventions).

---

## Gotchas & tricks

- **Subtler styles work better.** Extreme styles ("only emojis") often trigger safety retraining targeted at those specific patterns. Medium-distance styles ("casual slang with minor misspellings") are the sweet spot — far enough from refusal templates, close enough to natural language that the model doesn't notice the trap.
- **Hybrid with roleplay.** Style injection combined with a character ("as a 19th-century smuggler would phrase it") doubles as [roleplay-jailbreak](roleplay-jailbreak.md). The character provides justification for the style.
- **Length and form constraints.** "Write a haiku about X" is a style-injection variant — the haiku form has no room for a polished refusal, and the model either emits content-bearing lines or fails the format.
- **Mitigated by style-agnostic safety.** Some newer safety training explicitly adds refusals in many styles (pirate-speak, haiku, etc.) to the preference data. This narrows the gap but doesn't close the mechanism — novel styles keep appearing.
- **Evaluation pitfalls.** Graders (human or model-judge) need to extract the harmful content through the style. A pirate-speak description of how to build X is still "how to build X." Don't let style obscure the scoring.

---

## Sources

- Paper: *Jailbroken: How Does LLM Safety Training Fail?* — Wei, Haghtalab, Steinhardt, 2023, [arXiv 2307.02483](https://arxiv.org/abs/2307.02483) — style injection as a named component of combination attacks.
- Paper: *Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models* — Shen et al., 2023, [arXiv 2308.03825](https://arxiv.org/abs/2308.03825) — shows style-constrained prompts as features of high-ASR community prompts ("Narrative" / "Fictional" clusters).
