# HyperBrowseComp
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A 423-question hand-authored web-browsing benchmark across 13 languages, written by native/proficient speakers, stress-testing agents on multi-hop information retrieval across heterogeneous sources (videos, scanned documents, images, maps). Easier questions are filtered by evaluating candidates against internet-free models to guarantee that parametric knowledge alone cannot answer them. Introduced in *HyperBrowseComp* (2026, arXiv:2610.03574).

**Prereqs:** [README.md](README.md)
**Related:** [../agents/README.md](../agents/README.md), [../safety/low-resource-language-jailbreak.md](../safety/low-resource-language-jailbreak.md)

---

## What it is

A deliberately hard browsing benchmark. Every question targets a concise, publicly verifiable answer, but reaching that answer requires locating obscure evidence, following a multi-step clue chain, or inspecting non-text modalities (videos, scanned PDFs, images, maps). Questions span 13 languages (English and 12 others), each authored by a native or highly proficient speaker.

## How it works

Three design choices distinguish HyperBrowseComp from prior browsing evals (BrowseComp, SimpleQA):

1. **Multilingual by construction.** Each author writes in their native language; a bilingual verifier confirms the question-answer pair is well-formed before inclusion. English is one of 13 languages, not the default.
2. **Multimodal evidence.** A deliberate fraction of questions require evidence in non-text sources — a figure in a scanned paper, a frame in a video, a location on a map. Pure-text agents cannot cover these.
3. **Difficulty filter via no-internet baselines.** Candidate questions are run through strong models with internet access disabled. Questions that these models answer correctly from parametric knowledge are dropped. What remains is forced to depend on actual retrieval.

Evaluation uses provider-native search when available and a shared external retrieval harness as a control, both inside a common agent protocol. Human evaluation is run on a sampled subset to establish ceiling performance.

## Why it matters

Browsing agents have become a central product surface (Perplexity, ChatGPT Search, Gemini Deep Research). Existing benchmarks either saturate (SimpleQA) or are English-only (BrowseComp). HyperBrowseComp attacks the real deployment surface: cross-lingual, cross-modal, multi-hop information needs. Scores at launch show the gap between humans and frontier agents is wide — the benchmark is far from saturated.

## Gotchas & tricks

- **The no-internet filter has to be re-run periodically.** As models' parametric knowledge grows, previously-hard questions become easy; a filter refresh keeps difficulty calibrated.
- **Agent protocol decisions dominate scores.** Allowed tools, retries, retrieval depth, and prompt scaffolding each shift results. The authors ship a reference protocol; cross-paper comparisons require matching it.
- **Scanned-document questions reward specific tools.** Agents with OCR access score materially higher on the subset. Reporting a single aggregate obscures the capability axis.
- **The 13-language split is not evenly difficult.** Low-resource languages have fewer corroborating sources online; "same question, different language" can land very different difficulty.

## Sources

- Paper: *HyperBrowseComp: A Multilingual and Multimodal Stress Test for Web-Browsing Agents* — Aji, Ramadhan, Zuhri et al., 2026 — [arXiv:2610.03574](https://arxiv.org/abs/2610.03574).
