# Falsification Verifier (HTV-Agent)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A verifier-agent that accepts a synthetic-data answer only if multi-source counter-evidence searches *fail* to refute it. Inverts the usual "score the answer" framing into a Popperian falsification loop: every candidate (question, answer) starts as a hypothesis, and the verifier's job is to find evidence that breaks it. Introduced in VeriEvol (2026) for multimodal-math data; adds +2.06 over baseline at fixed RL conditions when paired with type-aware Evol-Instruct.

**Prereqs:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md)
**Related:** [evol-instruct.md](evol-instruct.md) · [../agents/execute-distill-verify.md](../agents/execute-distill-verify.md)

---

## What it is

Standard answer verification for synthetic-data pipelines runs a scorer model that says *yes* or *no*. Scorers share the generator's blind spots — both confidently agree on a wrong answer that "looks right." A falsification verifier flips the polarity: it actively tries to *refute* each candidate answer, and only the candidates that survive multiple refutation attempts are kept.

## How it works

For each candidate (question, answer) from the generator:

1. **Generate counter-hypotheses.** The verifier-agent (HTV-Agent in VeriEvol) generates plausible alternative answers or constraints the candidate must satisfy.
2. **Search for counter-evidence.** The agent queries multiple sources — different solvers, alternative reasoning paths, retrieved references, code execution — looking for evidence that refutes the candidate.
3. **Aggregate.** If any independent source produces a contradicting answer that itself survives basic sanity checks, the candidate is rejected. Only candidates that no source can refute are written to the dataset.

The pattern is structurally Popperian: an answer is provisionally accepted not because it is *proven* correct, but because no attempt to refute it succeeded. With diverse refutation sources, the survival rate is a defensible quality signal.

## Why it matters

- Catches a failure class scorers miss: confidently-wrong answers that share the generator's biases.
- The "multi-source" structure is the key — a single second-opinion model is barely better than the generator; diverse sources break correlated errors.
- Generalizes to any synthetic-data pipeline where answers can be probed (math, code, retrieval, structured extraction).

## Gotchas & tricks

- Source diversity is the whole game. If all "independent" sources are calls to the same model with different prompts, refutation correlates and the verifier is fooled.
- Recall vs. precision: aggressive refutation criteria reject valid answers; permissive ones admit bad ones. The paper recommends multi-round agreement thresholds.
- Expensive at generation time (multiple refutation attempts per candidate), but the resulting dataset is reusable.
- Distinct from outcome reward models (which score) — falsification verifiers refute. Distinct from EDV ([../agents/execute-distill-verify.md](../agents/execute-distill-verify.md)) which separates roles across an agent loop, not a data-generation pipeline (though the underlying defense is the same).

## Sources

- Paper: *VeriEvol: Scaling Multimodal Mathematical Reasoning via Verifiable Evol-Instruct* — Zheng et al., Tsinghua / Tencent Hunyuan, 2026 — [arXiv:2606.23543](https://arxiv.org/abs/2606.23543).
- Philosophical reference: Popper, *The Logic of Scientific Discovery* — falsificationism.
