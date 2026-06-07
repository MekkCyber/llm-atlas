# Memorization Propensity (PropMe)
*Depth — separate "can the model leak training data under attack" from "does the model leak in ordinary use."*

**TL;DR:** Memorization audits usually answer one question: with a prefix attack (feed the first N tokens of a training document, ask for the rest), how much can the model recite? That measures *capability*. The harder, more practically relevant question is *propensity*: how much does the model leak in *non-adversarial* use? **PropMe** (Barmina, Schneider-Kamp, Galke Poech, 2026) formalises the distinction with a metric transformation that converts any existing memorization score into its propensity counterpart, plus **SimpleTrace**, an infini-gram-based pipeline that deterministically attributes generations to large training corpora. On two fully open models (Comma, DFM Decoder) over Common Pile and Dynaword, the capability/propensity gap is large and consistent.

**Prereqs:** [safety/README.md](./README.md), [data/decontamination.md](../data/decontamination.md)
**Related:** [data/deduplication.md](../data/deduplication.md), [data/_data-curation.md](../data/_data-curation.md)

---

## What it is

A memorization metric $M(\text{prompt}, \text{generation})$ scores how closely a generation overlaps with a training corpus. Existing literature evaluates $M$ under *capability* prompting: e.g. take training documents, present a prefix, see if the model continues verbatim. This bounds the worst case but tells you little about ordinary use.

PropMe defines a transformation $T$ such that $T(M)$ measures memorization *under the model's natural prompt distribution*: generic prompts ("explain X"), dataset-specific prompts ("what does $X$ say"), no prefix at all. The capability score is an upper bound; the propensity score is the realistic exposure.

## How it works

### Metric transformation

For any base memorization metric $M$, define:

$$
\mathrm{propensity}(M) = \mathbb{E}_{\text{prompt} \sim P_\text{ordinary}} \big[ \, M(\text{prompt}, \text{model}(\text{prompt})) \, \big]
$$

The transformation $T$ is just *which prompt distribution you average over*. Capability metrics use a worst-case distribution (prefix attacks). Propensity metrics use an ordinary-use distribution (generic instructions, dataset-paraphrased queries). The same $M$ can plug into either.

### SimpleTrace

A tracing pipeline that, given a generation, finds the longest verbatim and near-verbatim matches against a training corpus using **infini-gram** indices (suffix arrays over the corpus). Outputs:

- longest verbatim match (any length);
- near-verbatim score (edit distance to closest training span);
- propensity-transformed counterparts.

Crucially, SimpleTrace is deterministic — no LLM judge — so the audit is reproducible.

### Comma vs DFM Decoder

The paper applies the framework to two open models. DFM Decoder is *continually pre-trained* from Comma on partly different data. PropMe shows that DFM Decoder's capability *and* propensity for Common Pile (Comma's training set) both **decrease** after continued training. Continued training partially forgets — a result that's interesting because it's measured at both the worst-case and ordinary-case ends.

## Why it matters

- **Sharper standard for any open release.** A two-number report (capability + propensity) is dramatically more informative than a single worst-case extractability score.
- **Aligns audits with realistic risk.** Privacy risk in deployment is set by ordinary leakage, not by adversarial extractability — unless your threat model includes a determined attacker with prefix access.
- **Provides an empirical handle on "forgetting."** Continual training as a memorization mitigation has been folklore; PropMe gives it a measurable trajectory.

## Gotchas & tricks

- **Prompt distribution is contestable.** What counts as "ordinary use"? PropMe uses generic and dataset-paraphrased prompts; production deployments will want their own ordinary-use distributions.
- **infini-gram needs the full training corpus indexed.** Only viable for open releases. Closed models can use a public-corpus approximation as a lower bound.
- **Verbatim vs near-verbatim matter differently for privacy and IP.** Verbatim is the canonical legal hook; near-verbatim is what most leakage actually looks like. Report both.
- **Doesn't cover *targeted* extraction.** PropMe is about average behaviour; for "given an attacker who knows what they want," extraction-attack literature (Carlini et al.) is still the right tool.
- **Continual training is not a free fix.** Forgetting one corpus may *increase* memorization for the new corpus; the gain is conditional on the data shift.

## Sources

- Paper: *LLMs Can Leak Training Data But Do They Want To? A Propensity-Aware Evaluation of Memorization in LLMs* — Barmina, Schneider-Kamp, Galke Poech (University of Southern Denmark), 2026 — [arXiv:2606.06286](https://arxiv.org/abs/2606.06286).
- Tool: *infini-gram* — used as the suffix-array index for SimpleTrace.
