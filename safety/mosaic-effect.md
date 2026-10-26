# Mosaic Effect

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A confidentiality-breach class specific to multi-document agentic LLM systems. The agent discloses a **TOP SECRET composite inference** that can be reconstructed by combining two or more documents each marked individually SECRET (or lower). No single source is leaked; what leaks is the *synthesis*. Standard output filters and refusal training don't close the hole because the sensitive artifact is produced inside the chain of thought from permitted inputs.

**Prereqs:** [_attacks](_attacks.md), [cot-monitoring](cot-monitoring.md)
**Related:** [refusal-suppression](refusal-suppression.md), [safety-case](safety-case.md)

---

## What it is

Classical document confidentiality is per-document: each source has a classification, and the system enforces access per source. The **mosaic effect** is the long-known intelligence-community observation that an aggregator with access to several individually-permitted sources can derive a conclusion more sensitive than any one of them.

LLM agents are, by construction, aggregators. The Agent Mosaic benchmark (Luceri & Ferrara, 2026) formalizes this: build minimal examples with two SECRET documents whose composition yields a TOP SECRET inference, and show that frontier models and off-the-shelf agent harnesses leak the composite on benign queries at non-trivial rates.

---

## How it works

### Attack construction

Each example is a triple (query, source A, source B) with:

- Both A and B marked SECRET; each individually releasable.
- A composite inference marked TOP SECRET that is derivable only when A and B are read together.
- A **benign query** that doesn't ask for the composite but makes it likely to surface during synthesis.

Example domains in the benchmark: cybersecurity (vulnerability + exposed asset ⇒ exploit path), medical (two patient-record fragments ⇒ identifying diagnosis), military (unit location + capability ⇒ operational plan).

### Why standard safeguards don't catch it

- **Output-side refusal classifiers** look for the sensitive statement in the model's own output. The composite is often not a verbatim string in either source; it's a derivation. A simple regex or classifier trained on verbatim leaks won't fire.
- **Prompt-level rules** ("don't reveal TOP SECRET info") give the model a definition without the ability to detect when it's producing a TOP SECRET inference from SECRET inputs.
- **Refusal itself can leak.** The paper documents cases where the refusal explanation (*"I can't tell you how X and Y combine because that would reveal …"*) names the forbidden composite in the explanation.

### Partial mitigations tested

- **Visible classification headers** in each retrieved document + **explicit aggregation rules** in the system prompt cuts pooled disclosure in the paper's setup from ~69% to ~32% — real reduction, still ~1 in 3 answers leaking.
- **Deterministic policy checks** at retrieval, tool use, and answer release: pointed to in the paper's conclusion as the only reliable mitigation, since they don't rely on the model's own ability to reason about aggregation.

---

## Why it matters

- **Dominant deployment pattern.** RAG apps, enterprise agents, and deep-research agents are all multi-document aggregators. Any production system that retrieves from a mixed-classification corpus is in scope.
- **Reframes agent safety.** Shifts the burden from "train the model to refuse" to "wrap the loop with deterministic checks" — a concrete architectural prescription that LLM-only guardrails can't satisfy.
- **Benchmark is small but minimal.** 30 constructed examples × 3 domains, run in simulation (10 800 answers) and under live agent harnesses (Pi, Hermes with GPT-5.6 Sol, 1 440 answers). Small enough that any new model can be evaluated cheaply.
- **Not solved by better models.** The headers+rules mitigation applies to every tested model and leaves a comparable residual leak rate — raw capability doesn't help.

---

## Gotchas & tricks

- **Refusal can leak more than compliance.** When testing a system, include refusal content in the leak audit, not just the body of answered responses.
- **Benign queries matter more than adversarial ones.** The authors deliberately use *benign* queries — the system fails even when nobody is attacking. Adversarial jailbreaks on top of mosaic examples are expected to be strictly worse.
- **Live > simulation on some domains.** Military disclosure rose in the live harnesses relative to the controlled simulation. Simulated audits under-estimate real deployment risk.
- **Deterministic checks need provenance.** You can't enforce "no aggregation of A+B" without knowing which chunks the agent read; retrieval traces must be first-class in the eval harness.

---

## Sources

- Paper: *Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems* — Luceri & Ferrara, USC Information Sciences Institute, 2026 — [arXiv:2610.30418](https://arxiv.org/abs/2610.30418).
- Precedent: intelligence-community literature on the "mosaic theory" of classified information (pre-LLM); the paper imports the term.
