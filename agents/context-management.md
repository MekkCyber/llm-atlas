# Context Management for LLM Agents
*Depth — what gets dropped from the rolling context window, when, and how to measure whether the model actually internalized it before the drop.*

**TL;DR:** Long-horizon agents compress, summarize, and evict tokens to stay inside a finite context window. The implicit assumption is that the model has *internalized* what's being dropped — but it usually hasn't. Mehta & Datta (2026) introduce *replay pairing*, a diagnostic that runs the same trajectory with and without a candidate piece of information in context and measures hidden-state divergence. On Llama-3.1-70B, plan signal collapses 4.1× one step after the plan is dropped; on HotpotQA, 12.4×. Naïve plan eviction costs 34.7 pp on ALFWorld and probe-gated re-surfacing does *not* recover it. Implication: context management is a load-bearing safety property of agent systems, not a UI nicety.

**Prereqs:** [_agent-memory](_agent-memory.md), [interpretability/logit-lens](../interpretability/logit-lens.md)
**Related:** [_post-training](../post-training/_post-training.md), [_attacks](../safety/_attacks.md)

---

## What it is

The set of policies an agent uses to keep its conversation / trace inside a finite context window:

- **Truncation.** Drop the oldest tokens.
- **Summarization.** Replace a span with an LLM-written summary.
- **Selective eviction.** Identify "low-value" content (tool outputs, repeated reasoning) and remove it.
- **Hierarchical buffering.** Move evicted content to an external memory tier and re-surface on demand.
- **Plan / scratchpad protection.** Pin the plan or scratchpad so it never gets evicted.

Each of these is correct *only when* the dropped information has either been internalized by the model (encoded into the action policy) or is no longer needed. The diagnostic question is: how do you know?

## How it works

The Mehta & Datta (2026) recipe answers the diagnostic question via **replay pairing**:

1. Take a complete agent trajectory $\tau$.
2. At a candidate eviction point, replay $\tau$ in two conditions:
   - **Kept:** the full context.
   - **Stripped:** the candidate information (e.g. the plan) removed.
3. At each subsequent step, compute the cosine distance between the hidden states under the two conditions, at a target layer.

**The reasoning-trace confound.** For R1-style reasoning models, `<think>` blocks earlier in the trace re-derive the plan content. Standard stripping leaves plan evidence in the stripped condition; the divergence looks small even when the plan really has been dropped. The fix is *strict stripping*: also remove prior `<think>` blocks from the stripped run, but not the kept run. This recovers +163 % of the step+1 signal in-sample and +153 % held-out.

**Probe transfer.** A layer-$L_{32}$ probe trained on Llama-3.1-70B transfers to DeepSeek-R1-Distill-Llama-70B at AUROC 0.748; R1-specific probes reach 1.000, suggesting R1 encodes plan signal in a different hidden-state direction.

**Stress test.** On ALFWorld, naïve plan eviction at compression time costs 34.7 pp success; probe-gated re-surfacing (drop the plan, re-insert when the probe detects context-only state) does *not* recover it. The conclusion is structural: plans are *context-resident*, not persistent.

## Why it matters

- **Reframes context management.** It's a load-bearing safety property: dropping the wrong span doesn't just lose tokens, it loses state the model never internalized.
- **Falsifies "compression is free."** Most agent frameworks compress aggressively; the paper provides a measurement showing the cost on plan-dependent tasks.
- **Suggests architecture, not just bookkeeping.** Probe-gated re-surfacing failing on ALFWorld implies the fix is upstream — either training the model to internalize plans, or persisting plans outside the context window (memory) rather than relying on context.

## Gotchas & tricks

- **Strict stripping for reasoning models.** Whenever you measure context dependence on R1-style models, strip prior `<think>` blocks from the stripped run only. Otherwise the measurement looks weaker than it is.
- **Layer choice matters.** Hidden-state cosine at the wrong layer hides the signal — pick a mid-network layer (the paper uses $L_{32}$ on 80-layer Llama).
- **Don't conflate "model uses plan" with "model needs plan in context."** The probe detects presence in residual stream; whether the action policy *uses* it is a separate causal question.
- **Generalization across tasks varies.** HotpotQA shows a 12.4× decay vs Llama's 4.1× — task-dependent.
- **Compose with memory.** Pair plan-protection with explicit external memory (see [_agent-memory](_agent-memory.md)) rather than relying on either alone.

## Sources

- Paper: *Plans Don't Persist: Why Context Management Is Load Bearing for LLM Agents* — Mehta & Datta (Snowflake AI Research), 2026 — [arXiv 2606.22953](https://arxiv.org/abs/2606.22953).
- Background: *MemGPT* — Packer et al., 2023 — for hierarchical memory tiers.
- Companion: *Are We Ready For An Agent-Native Memory System?* — OpenDataBox, 2026 — [arXiv 2606.24775](https://arxiv.org/abs/2606.24775) — taxonomy of external memory designs that compress-and-evict tries to substitute for.
