# Selective Non-Compliance in VLMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Existing VLM safety benchmarks treat each query as wholly comply-or-refuse. Real user queries mix answerable and non-compliant components. **KoNA** evaluates selective non-compliance — a VLM should answer the answerable portion of a mixed query while withholding compliance on the rest — across five categories: False Premise, Visual Inaccessibility, Universal Unknown, Task Feasibility, and Safety.

**Prereqs:** [README.md](./README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [refusal-suppression.md](./refusal-suppression.md), [rationale-only-safety-tuning.md](./rationale-only-safety-tuning.md)

---

## What it is

Current VLM safety evals label whole queries as either "should comply" or "should refuse." That framing hides the common real-world case where a single query mixes valid and problematic components: *"Describe the man's face and identify him from his driver's license."* The first half is fine; the second half should be withheld.

KoNA scores a VLM's ability to answer the answerable *and* withhold the rest — call it selective non-compliance — across five categories of non-compliance triggers, each of which can be entangled with legitimate content.

## How it works

The five categories:

- **False Premise.** The query asserts something the image does not show ("How many red cars are in this all-blue photo?"). The correct behavior is to correct the premise, then answer whatever remains.
- **Visual Inaccessibility.** The query asks about content the image occludes or crops out. Correct behavior: state what is not visible, answer what is.
- **Universal Unknown.** The query asks something no model could know (a stranger's private information, unlogged events). Withhold that part; answer the rest.
- **Task Feasibility.** The query mixes feasible and infeasible sub-tasks (e.g. count vs. read microscopic text). Complete the feasible; explain the infeasible.
- **Safety.** The query mixes benign and harmful sub-requests. Comply with benign, refuse harmful.

Each benchmark item is a query mixing at least one non-compliance trigger with at least one legitimate request. Scoring rewards answering the legitimate part and correctly withholding the rest — whole-query compliance and whole-query refusal both lose points.

## Why it matters

- **Matches how users actually ask.** Real queries are almost never single-purpose. A safety evaluation that scores only whole-query behavior systematically over-rates over-refusing and under-detects half-answered queries.
- **Exposes the training gap.** Safety tuning has been binary; selective compliance requires per-component decisions inside a single response. Points at a data / objective gap in current safety pipelines.
- **VLM-specific traps.** Visual Inaccessibility and False Premise are unique to multimodal — you cannot re-use LLM-only refusal benchmarks to cover them.

## Gotchas & tricks

- **Scoring is non-trivial.** Split-answer scoring requires per-span judgment; naive whole-answer classifiers under-detect partial compliance. Use LLM judges with explicit rubrics per KoNA category.
- **False Premise ⇄ hallucination.** A VLM that hallucinates the premise (invents the "red cars") loses on hallucination *and* selective compliance — cross-check with dedicated hallucination evals rather than conflating.
- **Refusal style masks behavior.** A model that refuses everything scores 0 on selective compliance despite looking "safe" on binary evals — a diagnostic feature, not a bug.
- **Category imbalance.** Safety-category prevalence in real traffic is likely different from KoNA's uniform sampling; re-weight for deployment-relevant baselines.

## Sources

- Paper: *Knowing What Not to Answer: Selective Non-Compliance in Vision-Language Models* — Minji Kim, Jihyoung Jang, Hyounghun Kim — POSTECH — arXiv:2609.04720 (2026).
