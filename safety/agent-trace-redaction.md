# Agent Trace Redaction
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Published agent execution traces — tool invocations, intermediate decisions, error-recovery branches — leak the underlying *procedural skill* of the agent. RedAct is a framework that **localizes** the secret-bearing tokens in a trace, **rewrites** them while preserving auditor-checkable evidence, and **watermarks** the result for provenance. Reduces normalized skill transfer from 44.7–67.1% on raw traces to below the no-skill baseline, with 93.6–100% watermark detection at ≤1.9% false-alarm.

**Prereqs:** [cot-monitoring.md](cot-monitoring.md), [_attacks.md](_attacks.md)
**Related:** [safety-case.md](safety-case.md), [../agents/agent-harness.md](../agents/agent-harness.md)

---

## What it is

Agents are increasingly *publishing* their execution traces — for trust (users see what the agent did), audit (operators check compliance), or training data (other agents learn from successful trajectories). The trace appears innocuous: tool calls, parameter values, intermediate text. But aggregated traces are sufficient to recover the agent's procedural skill — the formulas, thresholds, and strategy logic that distinguish it.

This is a *distillation channel without weight access*. A competitor doesn't need to steal model weights; they harvest published traces and SFT a small model on the recovered procedure. RedAct treats this as a security interface and provides selective redaction.

---

## How it works

### Threat model

Two parties: an honest *auditor* who needs to verify the agent acted correctly, and an adversarial *extractor* who reuses traces to clone the agent's skill. The desideratum: auditor evidence preserved, extractor signal suppressed.

### Localize → rewrite → watermark

1. **Localize.** Identify the trace tokens whose presence enables skill transfer. The paper uses an attribution score over a trace's contribution to a downstream "extract skill" probe; high-attribution spans are flagged.
2. **Rewrite.** Replace flagged spans with semantically equivalent but procedurally uninformative substitutes. Verifier-critical evidence (final outcomes, audit hooks) is *kept* by construction — the rewriter is constrained to leave those tokens untouched.
3. **Watermark.** Embed a behavioral watermark — a small statistical pattern in the rewritten trace — so that downstream uses are detectable. The watermark survives rewording but fails on traces that were generated independently.

### CapTraceBench

The companion benchmark: **75 long-horizon tasks × 154 curated skills × 7 domains**. Each (task, skill) pair allows measuring *normalized skill transfer* (NST) — how much of the target skill a downstream model can recover from a trace corpus. RedAct's primary metric is reducing NST without dropping audit-task verification rate.

---

## Why it matters

- **Frames published traces as a security interface.** Before this work, trace publication was treated as a transparency feature with no obvious downside. RedAct quantifies the downside.
- **Compatible with audit pipelines.** Auditor evidence is preserved by construction, so trust workflows don't need changes.
- **Watermarks support provenance enforcement.** Operators can detect when their traces are being reused outside permitted contexts.
- **NST below no-skill baseline** means the released traces are *less* useful for skill transfer than a random initialization — a strong protection.

---

## Gotchas & tricks

- **Attribution is the load-bearing step.** If the attribution score is wrong, RedAct redacts the wrong tokens. The paper's attribution uses a learned skill-extract probe; weaker proxies (token rarity, length) under-perform.
- **Watermark robustness vs. detectability.** Stronger watermarks survive more attacks but are easier for the extractor to detect and strip. The paper picks a behavioral watermark in the procedure-choice distribution; alternatives (token-level statistical marks) have different tradeoffs.
- **Doesn't help against weight theft.** RedAct protects published *traces*. If model weights are leaked, the skill is gone with them.
- **Compatible with CoT monitoring** ([cot-monitoring.md](cot-monitoring.md)) — the monitor sees the raw trace internally, the user sees the redacted version.
- **Auditor adversary attack.** An adversarial auditor with access to the rewriter could probe redaction boundaries. RedAct's protection is against passive trace reuse, not active inversion.
- **Re-training the agent invalidates watermarks** unless the watermark is rederived on the new policy — operationalize this in the deployment pipeline.

---

## Sources

- Paper: *RedAct: Redacting Agent Capability Traces for Procedural Skill Protection* — Xu, He, Fung, *et al.*, 2026 — [arXiv 2606.10813](https://arxiv.org/abs/2606.10813).
- Repo: [github.com/XuShuwenn/RedAct](https://github.com/XuShuwenn/RedAct).
- Background: model-extraction and watermarking literature; CoT monitoring as the on-line analogue.
