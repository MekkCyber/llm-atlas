# JetSpec
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A head-based speculative-decoding scheme that drafts an entire candidate tree in a *single* forward pass while keeping each branch *causally conditioned* on its parent path. Closes the gap between cheap-but-incoherent block-diffusion drafters and accurate-but-expensive autoregressive tree drafters, so larger draft budgets keep converting into longer accepted prefixes.

**Prereqs:** [_speculative-decoding.md](_speculative-decoding.md), [multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [mtp.md](../pre-training/mtp.md), [_moe.md](../architectures/_moe.md)

---

## What it is

Speculative decoding (SD) accelerates autoregressive LLMs by drafting many tokens with a cheap drafter and verifying them in one parallel pass of the target model. SD has hit a *scaling ceiling*: increasing the draft budget (tree size) only helps if (a) acceptance stays high and (b) draft overhead stays low.

Two flavors trade these off poorly. **Autoregressive drafters** condition each draft on the path so far — high acceptance, but drafting cost grows with tree depth. **Bidirectional block-diffusion drafters** emit all positions in one pass — cheap, but each position is marginalized so sibling tokens can be individually plausible yet mutually inconsistent, wasting verification budget. JetSpec is the head-based scheme that gets both: one-pass drafting *with* per-branch causal scoring.

## How it works

- A small parallel draft *head* is attached to a frozen target model and fed the target's fused hidden states.
- In one forward pass it emits a full token tree where each branch is scored under the target model's own autoregressive factorization (i.e. a branch's joint probability factorizes left-to-right along that branch).
- The verifier then runs the standard speculative-decoding accept/reject loop on the tree; high causal coherence per branch ⇒ long accepted prefixes per verification step.
- Training uses fused hidden states from the (frozen) target so the head learns the target's local distribution without retraining the backbone.

## Why it matters

- **Beats the scaling ceiling.** Up to **9.64× speedup on MATH-500** and **4.58× on open-ended chat** on dense and MoE Qwen3 models on H100, with further gains shown via vLLM integration under realistic serving load.
- Bridges a real production gap: block-diffusion heads saturate at small trees, autoregressive heads cost too much at large ones. JetSpec lets you *spend* a bigger draft budget productively, which is exactly what high-throughput serving wants.
- Drop-in with frozen target models — no need to retrain the backbone, which makes it deployable on existing checkpoints.

## Gotchas & tricks

- Acceptance is bounded by how well the head approximates the target's local distribution; mismatch on long-tail prompts still degrades speedup.
- Tree shape (fanout × depth) is a tuning lever — bigger trees aren't always better even with JetSpec; pick to match the verifier's parallel capacity.
- MoE backbones complicate hidden-state caching for the head; the paper shows it works on Qwen3 MoE, but expert-routing variance can hurt acceptance.

## Sources

- Paper: *JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting* — Hu et al., UCSD / Zhejiang / StepFun, 2026 — arXiv:2606.18394.
- Code: https://github.com/hao-ai-lab/JetSpec
