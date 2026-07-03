# Rubric-Based Evaluation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of scoring a model's response against one gold answer (holistic semantic match), decompose the ground truth into a **rubric of atomic clauses**: *must-be-present* facts the response has to cover, and *common-error* clauses it must not match. A **gated aggregator** zeros the score if any mandatory clause is missed. Extends IFEval's "verifiable constraints" idea to open-ended multimodal reasoning (PerceptionRubrics, 2026) and, by extension, any answer whose correctness is decomposable.

**Prereqs:** [README](README.md), [ifeval](ifeval.md)
**Related:** [mmlu](mmlu.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A rubric-based evaluation replaces holistic "did the model's answer match the reference?" scoring with clause-level auditing. For each item $(x, y^*)$ in the benchmark, the reference is expanded into:

- **Mandatory clauses** $M = \{m_1, \ldots\}$ — atomic facts the response must state or imply.
- **Common-error clauses** $E = \{e_1, \ldots\}$ — statements or omissions typical of wrong responses.
- **Optional / bonus clauses** $O = \{o_1, \ldots\}$ — nice-to-have details for tie-breaking.

The scoring gates on $M$ before rewarding anything else: missing a single mandatory clause zeros the score, regardless of surface fluency. Common-error clauses actively subtract.

PerceptionRubrics (2026) pairs 1,038 dense images with over 12,000 rubric clauses and demonstrates the gated protocol; the same shape applies to text-only reasoning, code review, medical reasoning, and any domain where correctness decomposes.

## How it works

### Building the rubric

Two paths in the literature:

- **Human authored** — annotators list the atomic facts a passing answer must include and the common wrong turns to watch for.
- **LLM authored + human validated** — a strong model drafts the rubric from the reference, humans validate. Cheaper at scale; sensitive to the drafter's biases.

Each clause is either a **verifiable proposition** (regex, string match, tool call) or a **judge query** (asks a scoring LLM: "does the response state that X?"). The judge is prompted per-clause, not once for the whole response — the atomicity is the point.

### The gated aggregator

$$
\text{score}(y) = \begin{cases} 0 & \text{if any } m \in M \text{ is not satisfied by } y \\ 
1 - \lambda \cdot |\{e : e \text{ matches } y\}| + \mu \cdot |\{o : o \text{ satisfied}\}| & \text{else}
\end{cases}
$$

The gate on $M$ is the key move: it prevents a fluent-but-omitting-a-critical-fact response from earning partial credit on the wrong axes.

### Correlation with humans

Papers report much higher rank correlation with human-perception judgements than holistic-VQA scoring — the mandatory-clause gate captures what humans actually check for.

## Why it matters

- **Benchmarks saturate; models don't.** Holistic scoring gives fluent-but-wrong responses partial credit, hiding real capability gaps. Rubric-atomic scoring re-exposes the gap.
- **Aligns evaluation with what humans verify.** People check answers by asking "did it mention X? did it avoid Y?" — a rubric encodes exactly that process.
- **Domain-transferable.** The pattern extends from multimodal perception (PerceptionRubrics) to any answer with decomposable correctness — math proofs (step-must-hold clauses), code review (must-flag / must-not-flag), medical reasoning (differential must include / must not exclude).

## Gotchas & tricks

- Rubric quality dominates results. A sloppy rubric produces a sloppy benchmark, gated or not.
- Judge model bias: if the judge is the same family as one of the models under test, its scoring of that model is biased. Cross-family judges or verifier ensembles help.
- The gate is unforgiving by design. For high-stakes settings that's the point; for exploratory evals, soften with a partial-credit tail for near-hits.
- Common-error clauses are hard to author before you have data — usually authored by mining wrong responses from a first pass, then formalised.
- Overfitting to the rubric: models fine-tuned on rubric feedback learn to state every clause verbatim; watch for degenerate fluency.

## Sources

- Paper: *PerceptionRubrics: Calibrating Multimodal Evaluation to Human Perception* — Peng et al., 2026 — [arXiv:2606.28322](https://arxiv.org/abs/2606.28322).
- Paper: *Instruction-Following Evaluation for Large Language Models (IFEval)* — Zhou et al., 2023 — verifiable-constraint scoring for instruction following; the closest existing analogue.
- Paper: *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — Zheng et al., 2023 — the judge-LLM apparatus rubric protocols reuse.
