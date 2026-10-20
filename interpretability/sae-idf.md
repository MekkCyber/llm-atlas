# SAE-IDF — SAEs as a Shared Coordinate System
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Treats a sparse autoencoder's feature space as a **shared coordinate system** for matching symbols across different symbolic systems. For each symbol, embed its native context through an LLM, SAE-encode the activations, weight features by rarity (an **IDF**-style term over feature usage), and rank candidates by cosine similarity in the IDF-weighted SAE space. Every match is traceable to the named features that drove it.

**Prereqs:** [sparse-autoencoder.md](sparse-autoencoder.md)
**Related:** [lexical-reference-frames.md](lexical-reference-frames.md)

---

## What it is

Different symbolic systems (knowledge bases, ontologies, schemas, codebase vocabularies) use different names for overlapping concepts. Matching symbols across systems is a bottleneck for integration, retrieval, and lightweight neuro-symbolic work. Hidden-state cosine similarity gives weak results because transformer activations are polysemantic and dominated by generic features (position, syntax).

SAE-IDF replaces the hidden state with its **sparse-autoencoder code** and *re-weights* that code by feature rarity. Two symbols match when they share rare, informative features, not when they both activate common ones.

## How it works

**Per-symbol encoding.**
1. Embed each symbol in its **native context** (a short sentence drawn from its source system).
2. Run the LLM forward and collect the chosen layer's activations at the symbol's token positions.
3. Encode with a pre-trained SAE to get a sparse code $f(a) \in \mathbb{R}^D$.
4. Average pool across the symbol's tokens to get one code per symbol.

**IDF weighting.** For each SAE feature $i$, estimate its activation frequency $p_i$ over a reference corpus. Weight each code coordinate by $\log(1/p_i)$ (standard IDF). This suppresses features that fire on almost everything.

**Matching.** For a query symbol $s$ and candidate pool $\{t_j\}$, rank by cosine similarity between IDF-weighted SAE codes. Return the top match (or top-$k$ with a threshold).

**Explainability.** The dot product can be decomposed per feature; the top contributing features name the match in human-readable terms ("both symbols fire on the same 'public-transit-station' feature").

## Why it matters

- **Concrete downstream use for SAEs.** Previously, SAEs were mostly interp artifacts; SAE-IDF turns them into a general representation layer for matching, retrieval, and alignment.
- **Beats plain hidden states consistently.** Across 12 tasks over six classes of symbolic system: 12/12 on Gemma-2-9B, 11/12 on Llama-3.1-8B, with gains **up to 45 points** on tasks like entity linking and schema matching.
- **Explainable by construction.** Every match is attributable to specific SAE features, which can be labeled from top-activating texts.

## Gotchas & tricks

- **SAE choice matters.** TopK SAEs give sharper codes than L1 SAEs for this use. Dictionary width $D$ should be large enough to resolve the concepts in the matching domain.
- **IDF corpus matters.** Compute $p_i$ on a corpus that's broad relative to the matching task. Using the task's own symbols as the IDF corpus trivializes the weighting.
- **Context framing matters.** Symbols in isolation activate generic features. Always embed in a short native-context sentence drawn from the source system.
- **Not a replacement for ontology alignment tools.** SAE-IDF excels when context is rich enough to disambiguate; for context-free symbol lists, classical lexical matching still wins.

## Sources

- Paper: *Sparse Autoencoders Are a Shared Coordinate System for Symbolic Matching* — Meng, Zhang, Han, Zhai, 2026 — [arXiv:2610.19636](https://arxiv.org/abs/2610.19636)
