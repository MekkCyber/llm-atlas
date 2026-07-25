# Evolving User Intent
*Depth — a benchmark wrapper that lifts static single-turn tasks into multi-turn conversations where the user's intent is disclosed incrementally, revised, and sometimes redirected mid-conversation.*

**TL;DR:** Static single-turn benchmarks measure whether a model can answer a fully-specified prompt. Real usage is nothing like that — users drip out their intent, change their minds, and correct earlier framings mid-conversation. **Evolving user intent** is a wrapper protocol that turns any existing static benchmark into a multi-turn evaluation with a scripted user whose intent evolves across turns, while preserving the original task's evaluation metric so drops isolate the intent-tracking failure mode. Across families, strong static performance does not transfer.

**Prereqs:** [../evaluation/README](README.md), [ifeval](ifeval.md)
**Related:** [../agents/README](../agents/README.md)

---

## What it is

A meta-benchmark: take any single-turn task with an existing evaluation metric and wrap it in a scripted user simulator that:

- **Discloses intent incrementally.** The full task specification is chunked into disclosures across turns.
- **Revises specifications.** Some turns modify a previously-stated constraint (a date, a preference, a scope).
- **Redirects.** A subset of tasks changes objectives mid-conversation, testing whether the model updates its target instead of anchoring on the initial framing.

The original metric is applied to the model's final answer against the *final* intent. Because the underlying task is unchanged, any drop from the static baseline is attributable to intent-tracking failure, not to task difficulty.

## How it works

1. **Wrap.** Convert each static prompt $p$ into a script $(u_1, u_2, \dots, u_T)$ of user turns that jointly convey $p$ but individually leak only a portion.
2. **Interact.** The model responds after each user turn; the scripted user's next turn is a fixed function of the model's prior response (allowing conditional revision — "actually, since you mentioned X, forget Y").
3. **Evaluate.** The task's original metric is applied to the model's answer under the final intent. Compare to the static-benchmark score to quantify the intent-tracking gap.

Because the wrapper preserves the evaluation protocol, existing benchmarks (IFEval, code generation suites, task-completion evals) become drop-in *evolving-intent* variants without new annotation.

## Why it matters

- **Documents a systematic invisible gap.** Consistent, cross-family drops appear when static benchmarks are lifted into the evolving-intent regime — including on tasks the same model handles cleanly when the intent is spelled out up front.
- **Cheap to deploy.** No new labels; any existing benchmark is a candidate.
- **Aligns eval with product usage.** Real users don't specify tasks statically. Evolving-intent scores are much better proxies for downstream collaborative-agent behaviour than static ones.

## Gotchas & tricks

- **Script realism matters.** A user simulator that revises too aggressively becomes adversarial rather than realistic. Calibrate revision rates against measured user studies where possible.
- **Redirection is different from revision.** Revision keeps the objective; redirection changes it. Score them separately — models fail asymmetrically on the two.
- **Original metric quality is inherited.** If the underlying benchmark is noisy, the evolving-intent variant is too. Prefer benchmarks with high-quality metrics.
- **Failure ≠ user confusion.** Models sometimes fail by *sticking to* the initial intent after a revision. That is a real intent-tracking failure, not a good behaviour disguised as bad.

## Sources

- Paper: *LLMs Get Lost in Evolving User Intent* — Tack, Laban, Neville (Microsoft Research), 2026 — [arXiv:2607.20734](https://arxiv.org/abs/2607.20734).
