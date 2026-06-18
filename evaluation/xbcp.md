# XBCP — Cross-Lingual BrowseComp-Plus
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A controlled cross-lingual deep-research benchmark that keeps the English question and answer space of BrowseComp-Plus fixed but **varies the language of the supporting documents**. Any drop in agent performance is attributable to cross-lingual retrieval or reasoning rather than surface-language confusion in the prompt — a clean ablation that exposes a large hidden weakness in frontier deep-research agents.

**Prereqs:** (none — extends BrowseComp-Plus)
**Related:** [../agents/README.md](../agents/README.md), [README.md](README.md)

---

## What it is

Deep-research benchmarks (BrowseComp, BrowseComp-Plus, GAIA-style web tasks) have been almost exclusively English-question over English-evidence. This hides a real failure mode: real users routinely want answers grounded in evidence written in other languages (regulations, primary sources, local news).

Most "multilingual" benchmarks confound surface-language confusion (the prompt itself is non-English) with cross-lingual retrieval (the evidence is in a different language). XBCP isolates the second axis.

---

## How it works

### Controlled design

- **Question:** in English (identical to BrowseComp-Plus).
- **Expected answer:** in English (identical).
- **Evidence corpus:** swapped to a non-English language per task.

The QA pair language is held constant; only the language of the supporting documents varies. This isolates whether the agent can retrieve, parse, and reason over non-English evidence to answer an English question.

### Two metrics decompose the failure

- **Retriever recall@k** in the target language — does the retrieval system surface the right document?
- **Agent answer accuracy** given the (possibly imperfect) retrieved evidence — can the agent extract and reason over the foreign-language content?

By measuring both, the benchmark separates retrieval failures from reasoning failures.

### Languages covered

A set of typologically diverse target languages chosen to stress different failure modes (script difference, morphology, tokenizer coverage). The paper reports performance per language with per-axis breakdowns.

---

## Why it matters

- **Exposes a hidden weakness.** Frontier deep-research agents and retrievers degrade sharply when evidence is in a non-English language; the gap is large enough to be a primary failure axis.
- **Right ablation shape.** Holding the QA pair language constant cleanly attributes the gap to cross-lingual capability rather than prompt-side confusion — something most "multilingual eval" benchmarks fail to do.
- **Slots in next to BrowseComp / BrowseComp-Plus** as a standard agent-eval addition, with no need for a new question / answer pool.

---

## Gotchas & tricks

- **Tokenizer coverage as a confounder.** Some performance drops are pure tokenizer issues (rare-language tokens explode the sequence length / hit out-of-vocab fallbacks). The benchmark exposes the drop; it doesn't disentangle tokenizer-vs-model.
- **Corpus availability bias.** For some languages the evidence corpus is smaller / lower quality than English, contaminating the apples-to-apples comparison. The paper notes this and provides per-language corpus stats.
- **Asks one direction only.** English-Q / non-English-E. The mirror direction (non-English-Q / English-E) is a separate evaluation worth running but not covered here.
- **Doesn't fix the gap.** XBCP is diagnostic, not prescriptive. Methods that reduce the gap (multilingual retrievers, translation in the loop) are evaluated against it, not provided by it.

---

## Sources

- Paper: *Beyond Monolingual Deep Research: Evaluating Agents and Retrievers with Cross-Lingual BrowseComp-Plus* — Yuheng Lu et al., Waseda / Northwestern / RIKEN AIP et al., 2026 — [arXiv:2606.15345](https://arxiv.org/abs/2606.15345).
- Background: *BrowseComp-Plus* — the monolingual benchmark XBCP extends.
