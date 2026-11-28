# DRBench Pro
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for deep-research agents that scores a report by its **coverage** of a reference Wikipedia-derived knowledge tree, decomposed into breadth (across directions) and depth (along chains). 144 cases across 24 domains and 44,102 rubric items. The top agent reaches 57.04% overall coverage, and rubric judgments agree closely with human annotators.

**Prereqs:** [../evaluation/README.md](../evaluation/README.md)
**Related:** [../evaluation/mmlu.md](../evaluation/mmlu.md), [../evaluation/ifeval.md](../evaluation/ifeval.md)

---

## What it is

Deep-research agents (Perplexity-style long-form researchers) produce long reports with claims and citations. Scoring them has been a mess — pure LLM-as-judge prompts are vague and have low inter-annotator agreement. DRBench Pro replaces freeform judging with **rubric coverage over a structured knowledge reference**: the agent's report is scored on how many rubric items (derived from a known-good knowledge subgraph) it covers with evidence.

## How it works

1. **Build reference.** Construct knowledge trees from Wikipedia — nodes are concepts, edges are typed relations. Each tree represents one coherent research area.
2. **Sample research queries.** Sample a subgraph and derive a natural-language research prompt whose answer should touch the subgraph's nodes and edges.
3. **Derive rubrics.** From the sampled subgraph, generate structured rubric items. A rubric item is an atomic, evidence-grounded claim ("X was founded in Y by Z").
4. **Score report.** Judge the agent's report against each rubric item — covered (with evidence) / covered without evidence / not covered. Aggregate into:
   - **Breadth score** — fraction of research directions touched at all.
   - **Depth score** — mean rubric coverage along each touched direction.
   - **Overall** — combined.

## Why it matters

Three properties previous deep-research evals didn't have:

1. **Reproducibility.** The reference is a fixed graph — rubrics are deterministic once the graph is fixed. Different judges score the same reports similarly.
2. **Failure-mode decomposition.** Breadth-vs-depth exposes "shallow but wide" (agent lists everything briefly) vs. "deep but narrow" (agent writes an essay on one subtopic) failure modes separately.
3. **High human agreement.** Rubric judgments agree closely with human annotators on the same reports — because rubric items are atomic and evidence-grounded, not holistic opinions.

## Gotchas & tricks

- Wikipedia-derived references inherit Wikipedia biases and omissions. Reports that cover a correct non-Wikipedia claim don't get credit.
- The "evidence-grounded" filter penalises confident-but-unsourced writing styles — some strong agents score lower here than on freeform judges.
- Rubric generation uses LLMs too; sanity-checking a sample of rubrics against the source subgraph is recommended before trusting a new domain.

## Sources

- Paper: DRBench Pro: A Knowledge-Centric Benchmark for Professional Deep Research Agents — Yan et al. (Zhejiang Univ.), 2026 — [arXiv:2611.34852](https://arxiv.org/abs/2611.34852)
