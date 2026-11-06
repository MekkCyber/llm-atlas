# MSafeTemp — Scaling Multilingual Safety via Template Augmentation
*Depth — template-extraction + culturally-localized slot-filling for multilingual safety training.*

**TL;DR:** LLM safety alignment is overwhelmingly trained in English, and multilingual safety is fragile as a result — attacks that refuse in English succeed once translated. **MSafeTemp** scales a multilingual safety corpus to **60 languages** by (1) extracting reusable *safe templates* from harmful English prompts, (2) translating the templates (not just the raw prompts), and (3) filling slots with **culturally localized** harmful and benign entities. Training on it improves cross-lingual refusal; English-only evaluation systematically overestimates multilingual safety.

**Prereqs:** [low-resource-language-jailbreak.md](./low-resource-language-jailbreak.md)
**Related:** [mismatched-generalization.md](./mismatched-generalization.md) · [_jailbreaks.md](./_jailbreaks.md) · [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

A **template + slot** representation of harmful prompts. Rather than translating a full attack string, factor it into:

- a **template** — the structural shell of the attack (phrasing, instruction form, context),
- **slots** — the harmful (or benign) entities the template references (a drug, a weapon, a public figure, a target).

Then scale across languages via (translate template) × (localize slots). The resulting corpus is tagged by language, slot type, and whether the filled instance is harmful or benign.

## How it works

**Template extraction.** Start with a corpus of known harmful English prompts. For each, identify entity spans (NER + manual curation) and replace them with slot tokens. The remainder is the template.

**Translation of templates.** Use a translation model to render each template in each of the 60 target languages. Translating *templates* (not filled examples) avoids leaking English-specific slot vocabulary.

**Localized slot filling.** For each (language, slot_type), curate a **culturally localized** list of harmful and benign fillers — the specific drugs, targets, figures, or practices that are plausible in that language's cultural context. The same template is then instantiated with both locally-harmful and locally-benign fillers.

**Benign controls.** Each harmful instance is paired with a benign-filler sibling (same template, benign slot). This creates aligned contrastive pairs the alignment loss can use.

**Safety training.** Fine-tune the model on the corpus with the usual refusal/compliance targets. Benchmark cross-lingual: attacks the English model refuses, does the translated instance refuse?

## Why it matters

Low-resource-language jailbreaks are a well-known hole: high-resource-language alignment doesn't transfer to low-resource languages, and translation alone doesn't close the gap. Template-based augmentation is the first recipe at 60-language scale, with evidence that **English-only red-teaming is systematically over-estimating multilingual safety**. Template factoring is also reusable — the templates generalize across slot sets, so a new attack type extends the corpus by adding templates, not by re-translating thousands of filled examples.

## Gotchas & tricks

- Template extraction is the bottleneck; low-quality templates propagate into every language.
- Localized slot lists require per-language curation — translated English slot lists re-introduce the problem (English drug names in Swahili prompts).
- Benign/harmful contrast depends on **cultural** harm, not literal translation — what's benign in one culture may be harmful in another; keep per-language labels, not one global label.
- Training on heavy template augmentation can over-refuse (false positives on benign prompts that happen to share template shape) — balance with large-scale benign data.

## Sources

- Paper: *How I Aligned Your Model: Scaling Up Multilingual Safety Training with Templates* — Neuwinger, Yukhymenko, Schlag, 2026 — https://arxiv.org/abs/2610.40516
