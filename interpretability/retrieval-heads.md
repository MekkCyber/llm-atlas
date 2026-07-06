# Retrieval heads (literal vs non-literal)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A small subset of attention heads carry most of a model's long-context retrieval behavior. Existing detectors reward heads whose *attended* source token literally matches the *generated* token — a **read-side / literal-copy** criterion. **LOCOS** (Logit-Contribution Scoring) is a **write-side** detector that scores each head by the projection of its OV-circuit output onto the answer-token unembedding direction, so it also catches heads that *synthesize* a non-literal answer. Ablating the top LOCOS heads collapses long-context QA while parametric recall stays intact.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [README](README.md)

---

## What it is

An attention head splits into two circuits (Elhage et al. framing):

- **QK circuit** — decides *where* to attend. Read-side.
- **OV circuit** — decides *what* to write into the residual stream once attended. Write-side.

A "retrieval head" moves information from an in-context source position to the current position. Prior work (Wu et al., needle-in-a-haystack style probes) identified retrieval heads by matching: at some source position $s$, the head's attention is high AND the generated token equals the source token at $s$. That's a **literal-copy** criterion — it captures heads that read *from* copy-relevant places but doesn't capture heads that *write* an answer synthesized from context meaning.

**Non-literal retrieval** — paraphrasing an answer, resolving a coreference, executing a hop — leaves the read-side criterion blind: the generated token does not textually match any source token.

---

## How it works

**LOCOS (Gema et al., 2026)** scores each head $h$ at layer $\ell$ by:

$$
\mathrm{LOCOS}(h, \ell) = W_U[y]^\top \cdot \mathrm{OV}^{(h, \ell)}(\text{needle}) - W_U[y]^\top \cdot \mathrm{OV}^{(h, \ell)}(\text{off-needle})
$$

where $W_U[y]$ is the unembedding row of the answer token $y$, and the two OV outputs are the head's contribution when attending to the needle vs an off-needle span. Both are gathered in **one forward pass** — LOCOS instruments the model with hooks that capture the OV contribution at the specified positions.

Ranking heads by LOCOS gives a per-model list of "top retrieval heads." **Mean-ablating** the top $k$ heads at inference measures how much the model relies on them.

---

## Why it matters

- **Localized long-context.** On Qwen3-8B / NoLiMa (a non-literal-retrieval benchmark), ablating **50 LOCOS heads drops ROUGE-L from 0.401 → 0.000**, while the strongest attention-based baseline still retains 0.292 after the same-size ablation. The same 50-head ablation on Qwen3-8B drops MuSiQue 0.55 → 0.08 and BABILong 0.62 → 0.20, while a random-heads control stays within 0.05 of baseline.
- **Retrieval-specific.** Parametric-recall probes and arithmetic-reasoning benchmarks stay near baseline under the same ablation — LOCOS heads are the *retrieval* subcircuit, not general reasoning.
- **Interpretability targeting.** If a safety analysis needs to reason about how a model uses long-context information, the write-side head set is the right target, not the read-side one.

---

## Gotchas & tricks

- **Needle vs off-needle framing.** LOCOS needs a task with a clear needle span. For open-ended long-context tasks, choosing the "off-needle" contrast matters — Gema et al. use context spans matched in length and position but with the needle removed.
- **Cross-family transfer.** The paper validates on Qwen3, Gemma-3, and OLMo-3.1. The identified head *sets* differ; the *methodology* transfers.
- **Doesn't identify the QK "where."** LOCOS says which heads write the retrieved content, not which positions they read from. Combine with attention-pattern analysis if you need both.
- **Ablation scope.** Mean-ablation (replace head output with its dataset mean) is what the paper uses. Zero-ablation is stricter but changes other statistics; be consistent.

---

## Sources

- Paper: *Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads* — Gema, Alex, Minervini, 2026 — [arXiv:2607.01002](https://arxiv.org/abs/2607.01002).
- Background: *A Mathematical Framework for Transformer Circuits* — Elhage et al., 2021 — QK/OV decomposition.
- Baseline: retrieval-head detection via literal-copy attention matching (Wu et al., 2024).
