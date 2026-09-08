# Rationale-Only Safety Tuning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Safety-tuning responses are typically shaped as `[boilerplate refusal] + [rationale]`. Training a model on the refusal boilerplate teaches it to lean on surface trigger words ("shoot", "kill"), which produces **false refusals** on benign look-alike queries. Dropping the boilerplate and training only on the rationale reduces false refusals while keeping true-refusal rates on genuinely harmful queries roughly constant. A data-side intervention, no new loss or algorithm.

**Prereqs:** [README.md](./README.md), [refusal-suppression.md](./refusal-suppression.md)
**Related:** [selective-non-compliance.md](./selective-non-compliance.md), [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

RLHF and safety-SFT datasets pair harmful queries with responses that begin with a boilerplate refusal ("I'm sorry, I can't help with that") and continue into a rationale ("…because it could cause harm"). The boilerplate provides a highly-frequent, semantically-thin cue: the model can minimize training loss by matching *the surface presence of alarming tokens* to *the boilerplate opener*.

That shortcut is exactly why safety-tuned assistants over-refuse on benign queries containing scary words: *"Where can I shoot a good photo?"* triggers the same boilerplate as *"How do I shoot someone?"*.

The intervention: decompose each training response into (i) boilerplate refusal and (ii) rationale. Train only on the rationale. The model now has to *explain* why a query is harmful — which forces it to model the query semantics rather than a lexical trigger.

## How it works

- **Decompose the safety dataset.** For each `(harmful query, response)` pair, split the response into a refusal-statement segment and a rationale segment. This can be done with LLM-based segmentation or with a template match against the boilerplate distribution.
- **Retrain only on the rationale.** Fine-tune the base model on `(query, rationale)` pairs, dropping the boilerplate entirely. Optionally prepend a short generic refusal at inference if a boilerplate opener is desired stylistically.
- **Evaluate on both axes.** True refusal rate on genuinely harmful queries (safety) and false refusal rate on benign look-alike queries (false-refusal).

## Why it matters

- **Cheap and reversible.** No new loss, no new optimizer, no new architecture; it's a dataset transform. Trivial to A/B against the baseline.
- **Points at a general RLHF pathology.** *The label was training the wrong function.* Any dataset where a large fraction of the label is boilerplate risks the same failure — likely relevant beyond safety (e.g. tool-use, chain-of-thought preambles).
- **Complementary to attack-side work.** Most safety literature studies *jailbreaks* (getting refusals when you shouldn't). This paper is the flip side: unwanted refusals when you should get help — a top user complaint that has been undermeasured.

## Gotchas & tricks

- **Segmentation quality matters.** A noisy boilerplate/rationale split silently leaks boilerplate into the rationale training set and re-introduces the shortcut. Prefer conservative segmenters that keep only the confidently-classified rationale span.
- **Safety may still regress slightly.** The paper reports comparable, not identical, safety scores; deployments should re-check on their own harmful-query suites.
- **Composability with jailbreak defenses.** Rationale-only tuning doesn't fix specific jailbreaks (prefix injection, roleplay). Stack with dedicated defenses.
- **Style handoff.** Without boilerplate the model may respond in wildly varying tones. If a consistent refusal voice matters (product / brand), pair with a light style prompt.

## Sources

- Paper: *Refuse without Refusal: A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models* — Minji Kim, Hyounghun Kim — POSTECH — arXiv:2609.04714 (2026).
