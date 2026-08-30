# Claim Replay: Beyond Metric Reproduction in LLM Evals
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An eval artifact (task + scorer + reported metric) specifies a *forward computation*, but that computation does not necessarily **license the claim** attached to the metric — replaying the claim also needs bound historical evidence and semantic grounding that the eval often doesn't carry. Formalizing this "claim-replay layer" as a quadruple $(D, F, q, \text{identified set})$ turns eval hygiene from "the number reproduces" into "the number's *meaning* replays." A commit-bound census of Inspect Evals finds that **110 of 124 mechanically eligible units** cannot deterministically replay their claim.

**Prereqs:** [README.md](README.md)
**Related:** [../data/decontamination.md](../data/decontamination.md), [../safety/README.md](../safety/README.md)

---

## What it is

A methodology for auditing whether an eval's reported number *supports the claim it appears to support*. Forward reproducibility (same code + same prompt → same metric) is not enough: two evaluators can implement the same task and scorer and produce identical numbers whose intended meanings differ, because the historical evidence and alternative semantics required to *interpret* the number are unbound.

## How it works

The audit works over a formal quadruple, evaluated at a pinned commit:

- **$D$** — the frozen substrate: the exact inputs, prompts, scaffolds, and stopping rules used.
- **$F$** — a grounded family of admissible evaluator implementations (what counts as "the same scorer" or "the same protocol").
- **$q$** — the claim query: what the reported number is supposed to *mean* (a capability claim, a robustness claim, a relative claim between models).
- **Identified set** — the set of admissible answers to $q$ under $D$ and $F$. If it is a singleton, the claim is licensed by the eval; if it is empty or ambiguous, it is not.

**Typed stops.** Rather than forcing every eval into "robust / not robust," the audit records **why** a claim fails to replay: unbound historical evidence, ambiguous scorer family, absent semantic grounding, incompatible protocol variants, etc. Instability witnesses and stable substructure are recorded alongside.

## Why it matters

**110 of 124 mechanically eligible Inspect Evals units stop before deterministic inference** at a pinned commit — the metric computes, but the evidence needed to *interpret* it is missing or ambiguous. Where inference does close, exact values, winners, complete orders, and pairwise relations separate cleanly by claim resolution and by primary vs. review family.

Uncomfortable news for the safety-eval community: the number attached to a benchmark result is often a valid computation of an under-specified claim. The fix isn't "reproduce harder" — it's to make the claim-replay layer explicit, then audit it.

## Gotchas & tricks

- **"Robust" is not a scalar.** The audit deliberately refuses a single robust/not-robust label per eval — the useful output is the typed stop, not a badge.
- **Pinned commit matters.** Same eval at different commits can flip claim-replay outcomes without any metric change.
- **Downstream implication for leaderboards.** A leaderboard whose ordering depends on unlicensed claims cannot be defended by pointing at reproduced numbers.
- **Not just a paper's problem.** Any eval whose scorer, protocol, or historical evidence is under-specified inherits the same weakness — worth an internal audit before publishing a claim.

## Sources

- Paper: *What Does an Evaluation License? A Commit-Bound Census of Claim-Relative Inference in Inspect Evals* — Xi Qin, 2026 — [arXiv:2608.19269](https://arxiv.org/abs/2608.19269)
