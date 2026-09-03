# AgentJudgeBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A 3,808-instance benchmark for **LLM-as-a-judge reliability on agentic tool-calling workflows** structured as DAGs — a setting distinct from open-ended text or preference judging. Six DAG topologies × three difficulty tiers, five generators (3B–70B + GPT-5.4), six judges (20B → frontier), paired with/without ground truth. Finding: on hard queries without ground truth, all six judges converge to a **77–82% alignment ceiling regardless of scale**.

**Prereqs:** none
**Related:** [ifeval.md](ifeval.md), [../post-training/_rewards.md](../post-training/_rewards.md)

---

## What it is

LLM-as-a-judge evaluation is standard for open-ended generation. AgentJudgeBench is the first benchmark to systematically stress it on **structured, dependency-driven** agentic outputs — specifically, tool-calling traces shaped as workflow DAGs, where correctness depends on argument fidelity and dependency preservation across nodes, not on prose quality.

The benchmark grid:

- **Instances:** 3,808 total.
- **Structure:** 6 DAG topologies × 3 difficulty tiers.
- **Generators:** 5 (open-weight 3B, 8B, 32B, 70B + GPT-5.4).
- **Judges:** 6 (20B → frontier scale).
- **Conditions:** paired with-ground-truth and without-ground-truth.

## How it works

Each instance presents a judge with an agent's tool-calling trace and asks whether it matches the target workflow. The paired conditions isolate the marginal effect of ground-truth exposure. Judges are scored on **alignment with the programmatic reference**; a separate human-validation study checks alignment with human raters.

Mitigation strategies studied: chain-of-thought, judge temperature, structured evaluation rubrics.

## Why it matters

The paper's headline findings:

- **Monotone degradation with difficulty.** Judge alignment drops as task difficulty rises, and **1.5× faster without ground truth**.
- **Structural ceiling at 77–82% on hard queries.** Without ground truth, all six judges — 20B to frontier — collapse into the same 77–82% band. **Scale alone cannot break through this ceiling.** The ceiling is set by task difficulty and prompt design, not judge capability.
- **Ground-truth exposure is not uniformly helpful.** For GPT-5.4 and Gemini-2.5-Pro alignment *drops* by 1.5pp and 3.9pp respectively when the reference is exposed — consistent with over-anchoring.
- **Rubrics help, other mitigations don't.** CoT and temperature: negligible. Structured rubrics: up to +6.5pp, but not uniform across judge-generator pairs.
- **With ground truth, QwQ-32B best matches the programmatic reference.** In the human-validation study, GPT-OSS-120B is the most human-aligned judge.

Implications:

- **Agent-loop reward models inherit this ceiling.** Any RL loop using an LLM-judge reward on structured agentic outputs has a bounded reliability that model capacity cannot fix.
- **Prompt engineering (rubrics) beats scaling** for judge reliability, up to a point.
- **Concrete practitioner guidance** for reliable LLM-judge evaluation of tool-calling systems.

## Gotchas & tricks

- **The ceiling is per-benchmark, not universal.** 77–82% is specific to AgentJudgeBench's hardest tier; easier tiers have higher ceilings. Don't over-generalize the number.
- **Ground-truth over-anchoring is model-specific.** Not all judges are hurt by exposure — the effect is most pronounced on stronger judges, likely because they'd otherwise reason more independently.
- **Rubric gains don't compose.** Different judge-generator pairs benefit from different rubrics; there's no single rubric that helps everywhere.
- **Human-vs-programmatic alignment can diverge.** The best programmatic-alignment judge (QwQ-32B) is not the best human-alignment judge (GPT-OSS-120B). Pick the reference deliberately.
- **Judge temperature 0 is not automatically best.** The paper reports temperature had negligible effect over the range tested — not that any specific setting was optimal.

## Sources

- Paper: *AgentJudgeBench: A Multi-Difficulty Benchmark for Evaluating LLM Judges on Agentic Tool-Calling* — Verma et al., 2026 — [arXiv:2608.26623](https://arxiv.org/abs/2608.26623).
