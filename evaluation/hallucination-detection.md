# Hallucination Detection (Enoki-style Extract–Verify–Project)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** **Enoki** detects LLM hallucinations at both **claim** and **span** granularity in a single pipeline: extract text-anchored relational facts from the answer, verify each fact against a source, and project unverified facts back onto the offending answer spans. Modular back-ends (LLM-based, encoder-based, rule-based) trade cost for accuracy. Shipped with **EnokiQA**, 3,990 examples with aligned claim- and span-level labels.

**Prereqs:** [README.md](./README.md)
**Related:** [../safety/README.md](../safety/README.md)

---

## What it is

Hallucination detectors have historically split into two camps:

- **Claim-level.** Extract atomic claims from an answer, verify each. Interpretable but coarse (a whole claim is right or wrong, no span attribution).
- **Span-level.** Score every token / span in the answer for factual support. Fine-grained but hard to explain.

Enoki collapses them via **Open Information Extraction (OpenIE)**: it extracts relational facts *anchored to the answer's text spans*, verifies each fact against a source, then projects unverified facts back onto their source spans — so a single pass yields both claim-level verification decisions and span-level highlighting.

## How it works

- **Extract.** OpenIE over the model's answer produces `(subject, relation, object, [spans])` tuples where `spans` records which characters of the answer supplied each argument.
- **Verify.** Each tuple is checked against the reference source. Three swappable back-ends:
  - **LLM-based** — highest accuracy, slowest, most expensive.
  - **Encoder-based** — pretrained NLI-style entailment model; middle cost/accuracy.
  - **Rule-based** — string / structured-match rules; cheapest, coarsest.
- **Project.** Every tuple that fails verification propagates its `spans` back onto the answer text as a hallucination highlight. This is the step that yields span-level output from claim-level checks.
- **EnokiQA.** 3,990 examples labeled at *both* granularities with aligned annotations, letting the community evaluate the projection step directly.

## Why it matters

- **Unifies previously separate evaluation regimes.** One artifact for both claim-level and span-level hallucination benchmarks.
- **Cost/accuracy dial.** Deployers pick a back-end for their budget; the pipeline is the same. Natural fit for RAG systems that already produce `(answer, source)` pairs.
- **Interpretable by construction.** The tuple's spans are the explanation — no separate saliency method required.

## Gotchas & tricks

- **OpenIE quality is the ceiling.** Missed relational tuples become undetected hallucinations, whatever the verifier says. Prefer newer LLM-based OpenIE for open-domain answers.
- **Projection is not a saliency map.** The projected spans mark *where the unsupported fact was said*, not necessarily where the model went wrong internally.
- **Verifier hallucinations.** LLM-based verification can itself hallucinate; cross-check with encoder or rule-based back-ends on a diagnostic slice.
- **Answers without sources.** For no-source ("closed book") questions, the verifier back-end has to become a knowledge model — resurrect the exact problem you were trying to solve. Enoki is aimed at grounded answers.

## Sources

- Paper: *Enoki: Efficient Multi-Level Hallucination Detection* — Ionov, Ivanov, Savkin, Makarenko, Panchenko, Konovalov, Belikova — AIRI / Skoltech / ITMO / Sber AI Lab — arXiv:2609.00581 (2026). Ships EnokiQA.
