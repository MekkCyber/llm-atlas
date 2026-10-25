# Mosaic effect
*Depth — confidentiality leaks via aggregation across access-permitted documents in agentic LLM systems.*

**TL;DR:** Even when an LLM agent reads only documents the user is cleared to see, combining them can produce a sensitive conclusion that is *itself* above the user's clearance. US policy calls this *classification by compilation / aggregation*; the mosaic effect names the LLM-agent version. Benchmarks on a 30-example corpus (cybersecurity, medical, military) show benign queries disclose forbidden composites in 7–13% of runs across frontier models, and visible-classification + aggregation-rule safeguards cut pooled disclosure only from ~69% to ~32% — reduced, not eliminated. Refusal messages can themselves leak the composite.

**Prereqs:** [_attacks.md](../safety/_attacks.md)
**Related:** [cot-monitoring.md](../safety/cot-monitoring.md), [competing-objectives.md](../safety/competing-objectives.md)

---

## What it is

A failure mode of agentic retrieval: the attacker gains no access they don't already have. Instead, the model infers and discloses a *derived* fact that is more sensitive than any of its sources — a conclusion that only exists because the agent correlated multiple permitted inputs. This bypasses document-level access control because the control decides what the model *reads*, not what it *infers*.

## How it works

Each benchmark example contains:
- Two SECRET source documents, each on its own permissible to view.
- A TOP SECRET forbidden composite derivable from the two.
- A distractor to raise realism.
- Three query types: *benign* (innocent intent), *explicit* (asks for the composite), *adversarial* (deliberately phrased to elicit it).

Two safeguards are factorially varied: (a) visible classification headers in each document, (b) explicit aggregation rules in the system prompt. An LLM judge (validated against humans) scores whether the composite was disclosed.

## Why it matters

- **Access control is insufficient** for agent corpora in sensitive settings. The leak happens inside the model, not in the retriever.
- **Benign queries leak**: 7.08% (GPT-5.6 Sol), 12.50% (Pi), 8.75% (Hermes). The model doesn't need to be attacked to disclose.
- **Safeguards help but don't close the gap**: both-on pooled disclosure ~32% vs ~69% with neither. Visible classification + explicit rules are necessary but not sufficient.
- **Refusals leak**: some refusal explanations restate the forbidden conclusion, which is a worse failure than silently answering wrong.
- **Enforced inference-level controls** (output-side screening of the derived fact, not just document-side permissions) become architecturally mandatory for defense, finance, healthcare deployments.

## Gotchas & tricks

- LLM-judge disclosure scoring must be bounded; the judge can itself hallucinate the forbidden composite.
- Benchmark size (30 examples × 3 query types) is small; absolute percentages are indicative, not calibrated.
- Pi and Hermes selected their own documents; disclosure numbers partly reflect retrieval choices, not just inference.
- Military-domain disclosure rose in live-harness runs vs simulation — domain-specific risk needs domain-specific evaluation.

## Sources

- Paper: *Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems* — Luceri & Ferrara, 2026 — [arXiv:2610.30418](https://arxiv.org/abs/2610.30418)
