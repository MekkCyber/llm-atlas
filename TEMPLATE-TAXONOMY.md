# {{Category Name}}

*Taxonomy — {{one-line description of the class}}.*

**TL;DR:** one paragraph. What problem this class solves, the shape of the space, what the modern default is.

**Related taxonomies:** [_other-taxonomy](../path/_other-taxonomy.md)
**Depth files covered here:** [technique-a](technique-a.md) · [technique-b](technique-b.md) · …

---

## The problem

Why this class of techniques exists. What goes wrong without them. The underlying constraint every variant is fighting. One short section — don't over-explain.

## The shared pattern

What all variants have in common. The structural anatomy that makes them the same *kind* of technique. Name it concretely so the reader can see the category rather than a list of unrelated tricks.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [technique-a](technique-a.md) | one-line idea | one-line tradeoff | one-line use case |
| [technique-b](technique-b.md) | … | … | … |
| technique-c (no depth file yet) | … | … | … |

Link techniques with a depth file; leave others as plain text until a depth file lands.

## How to choose

Opinionated guidance. State the modern default plainly. Explain when you'd pick each non-default and why. If variants combine (e.g. stack multiple normalizations), say so here.

## Adjacent but distinct

Categories that look similar but belong in a different taxonomy. Link out with one-line reasons. Helps readers disambiguate.

## Sources

- Survey or overview paper — authors, year — link.
- Foundational references for each major variant, if not already cited inline.

---

## Conventions

- **Filename:** `_<category>.md` (leading underscore distinguishes taxonomy from depth files in the same folder).
- **Folder placement:** same folder as the depth files it links to.
- **Linking:** every depth file this taxonomy covers should reciprocally link back via its `Related:` line.
- **Scope:** keep one taxonomy per *class of techniques*. If a taxonomy is growing past ~8 variants with meaningfully different tradeoffs, consider splitting into sub-taxonomies.
