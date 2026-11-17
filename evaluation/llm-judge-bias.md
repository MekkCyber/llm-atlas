# LLM-as-judge bias (appendix bias and friends)
*Depth — quantified biases in LLM reviewers / judges, with the appendix bias as a concrete case.*

**TL;DR:** LLM reviewers systematically over-weight dense technical detail at the back of a paper. In an analysis of 2,000 papers from ACL, EMNLP, COLM, and ICLR, the latest GPT and Claude models draw roughly half or more of their identified weaknesses from the appendix, across API, coding-agent, and web access methods. Human reviewers pull 8.4% from the appendix and focus on presentation and the core idea. The bias misallocates reviewer attention to non-central details.

**Prereqs:** [./README.md](./README.md)
**Related:** [../safety/README.md](../safety/README.md)

---

## What it is

A measured bias class for LLM-as-judge configurations. LLM reviewers and judges are increasingly used for peer-review triage, RLAIF, and agent evaluation. This paper pins down a specific failure mode: positional / section-type bias in where critiques are sourced.

## How it works

Methodology:
- 2,000 papers from ACL, EMNLP, COLM, ICLR (mix of accepted / rejected).
- Latest GPT and Claude models prompted to produce reviews with identified weaknesses.
- For each weakness, label its source location (main body section vs appendix).
- Three access methods tested — plain API call, coding-agent access (shell, tools), web-based access. Same bias persists across all three.
- Compared against human reviewer traces where source locations are annotated.

Findings:
- **~50%+** of LLM weaknesses come from the appendix.
- **8.4%** of human weaknesses come from the appendix; human concerns cluster on presentation and the core idea.
- LLM critiques concentrate on fine-grained technical details that don't drive paper acceptance.

## Why it matters

LLM-assisted peer review is in active use at major venues. If LLM reviewers systematically pull attention to the appendix, they move the review signal away from the work's main contribution. The pattern reproduces across model families, access methods, and conferences — it's a model-level behavior, not a prompt artifact.

Short-term mitigations suggested by the result:
- Restrict the appendix from reviewer context, or weight main-body concerns higher.
- Instruct reviewers to summarize the main contribution *before* finding weaknesses.
- Treat LLM reviews as *technical linters* (useful on appendix detail, unreliable on paper-level judgment) rather than drop-in replacements.

## Gotchas & tricks

- **Not fixable by better prompting alone.** Three access methods all show the bias; prompt-only mitigations are partial.
- **Reviewer vs judge.** The study focuses on peer review but the same bias likely appears in any judge configuration where long context has an "appendix-like" tail — RAG with long retrieved passages, LLM-judge over long rollouts, agent-trace evaluation.
- **No evidence** of a compensating strength — the appendix critiques aren't more accurate, just more frequent.

## Sources

- Paper: *AI Reviews Are Biased Towards the Appendix* — Ghosh, Saim, Jiang (U. Cincinnati), 2026 — [arXiv:2611.17811](https://arxiv.org/abs/2611.17811)
