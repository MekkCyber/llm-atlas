# Unusual-Format Jailbreak

*Depth — ask for the response in a document format that safety training rarely used — JSON, poetry, song lyrics, screenplay — so the refusal templates don't fit.*

**TL;DR:** Wrap the harmful request with a format constraint — "Respond as JSON with a 'steps' array," "Answer as a rhyming poem," "Write as song lyrics with three verses." The model's trained refusals are in English chat-style prose. Unusual formats are well within its *capability* (it can write poetry, emit JSON), but safety training rarely covered refusing-in-format. The model fills the format with on-topic content because refusing *in-format* would be both awkward and outside its training distribution. A [mismatched-generalization](mismatched-generalization.md) attack on the format axis — distinct from [character-encoding-obfuscation](character-encoding-obfuscation.md) (which attacks the input side) and [style-injection](style-injection.md) (which is a competing-objectives attack on the refusal vocabulary).

**Prereqs:** [mismatched-generalization](mismatched-generalization.md)
**Related:** [character-encoding-obfuscation](character-encoding-obfuscation.md), [wikipedia-framing](wikipedia-framing.md), [style-injection](style-injection.md), [_jailbreaks](_jailbreaks.md)

---

## What it is

Constraints on the **output format** of the response. Common variants:

- JSON with a specified schema ("steps": [...], "materials": [...])
- YAML / XML / Markdown table
- Rhyming poem, haiku, limerick
- Song lyrics with verses, chorus, bridge
- Screenplay or dialogue form
- Recipe format (ingredients / steps / notes)
- Code comment (as if documenting a function that does X)

Wei et al. (2023) report a `few_shot_json` row showing high ASR on GPT-4 (0.53). Other format-constrained attacks appear across community jailbreak collections — notably within Shen et al.'s "Narrative" and "Fictional" jailbreak communities.

---

## How it works

### Format is a distributional axis safety didn't cover

Safety training operates mostly in conversational English. The preference RM scored refusals that looked like *"I'm sorry, I can't help with that."* It never scored:

- A JSON object whose `"answer"` key is "I cannot provide this."
- A haiku that refuses in 17 syllables.
- A screenplay where the assistant character declines.

When the user specifies a format the refusal template doesn't fit, the model has three options:

1. **Break the format.** Return prose refusal. Instruction-following violation.
2. **Refuse in-format.** Produce a JSON/haiku/screenplay refusal. Out of distribution — rarely fluent.
3. **Fill the format.** Produce on-topic content. Jailbreak.

RLHF has densely trained (1) against the user's format constraint; (2) was barely trained at all. (3) is the path that satisfies instruction-following with the lowest deviation from training distribution. Safety would have to aggressively override both — and the model, lacking the training to refuse-in-format, tends not to.

### JSON specifically is load-bearing in combination attacks

JSON-format attacks are unusually effective on GPT-4 for a specific reason: the model was fine-tuned on a large number of JSON-structured API responses (for tool use, function calling). The preference distribution for "valid JSON that answers the user's schema" is very sharp — the model strongly prefers filling the schema correctly over returning an error. When that schema happens to be `{"steps_to_harmful_thing": [...]}`, the JSON-formatting pressure fights refusal.

### Typical empirical numbers (Wei 2023)

`few_shot_json` ASR on GPT-4: **0.53**. Comparable to the strongest single attacks (AIM roleplay 0.75, evil_system_prompt 0.53, wikipedia_with_title 0.50). Format-constrained attacks are not the strongest in isolation, but they're load-bearing in combinations because they're structurally different from the competing-objectives layer.

---

## Why it matters

- **Attacks a dimension that surface filters miss.** Input-side filters looking for harmful keywords often miss the harmful *request for a format*. By the time content is generated, it's inside a JSON schema the filter may not inspect.
- **Exploits the strongest features of modern chat models.** Better format compliance is a selling point of instruction-tuned models. This same training makes them *more* vulnerable to format-based attacks, not less — another instance of mismatched generalization growing with capability.
- **Common in the wild.** Community jailbreak prompts frequently use output-format constraints to break past refusal templates, especially in code-formatted ("pretend you're documenting a function called `makeBomb(ingredients)`…") or creative-writing formats.

---

## Gotchas & tricks

- **Format must be plausibly benign.** JSON with arbitrary schemas is normal. Obscure formats invite the "you're trying to bypass safety" response. Keep the format request within the space of things the model was trained to follow.
- **Schema specificity matters.** A specific schema ("steps": [...] with 5 required entries) applies more structural pressure than a loose one. Model fills specific schemas more reliably.
- **Combines with character-encoding.** "Encode the response in Base64, then format it as JSON with a single 'data' field" stacks encoding and format attacks on different axes.
- **Narrative formats (story, screenplay) overlap with roleplay.** A request like "Write a scene in a screenplay where the villain explains how X" combines unusual-format with [roleplay-jailbreak](roleplay-jailbreak.md). The two mechanisms cover each other's weaknesses.
- **Some formats encourage structured harmful content.** JSON with `"steps": [...]` directly incentivizes structured step-by-step output, which is exactly what many harmful requests want. Rhyming poetry, by contrast, often dilutes harmful specifics via the rhyme constraint. Choose the format for what output shape the attacker wants.
- **Not the same as format-as-competing-objective.** If the format makes refusal *stylistically* impossible (pirate-speak, haiku), that's closer to [style-injection](style-injection.md) and exploits competing objectives. If the format is one the model can follow but safety wasn't trained on (JSON, screenplay), it's mismatched generalization.

---

## Sources

- Paper: *Jailbroken: How Does LLM Safety Training Fail?* — Wei, Haghtalab, Steinhardt, 2023, [arXiv 2307.02483](https://arxiv.org/abs/2307.02483) — reports `few_shot_json` ASR 0.53 on GPT-4; also evaluates poem and other format-constrained attacks as combination components.
- Paper: *Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models* — Shen et al., 2023, [arXiv 2308.03825](https://arxiv.org/abs/2308.03825) — "Narrative" and "Fictional" jailbreak community clusters frequently use format constraints.
