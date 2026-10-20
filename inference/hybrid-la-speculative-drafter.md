# Hybrid Linear-Attention Speculative Drafter (ReaDy)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Speculative decoding tailored to **hybrid linear-attention** LLMs. Each draft position uses a **learned query** to retrieve context from the target model's committed **recurrent states**; a lightweight *state reader* folds that retrieval into draft-space features. Reported 3.16× average decoding speedup on Qwen3-8B hybrid-LA, with no accuracy loss across seven tasks.

**Prereqs:** [attention.md](../fundamentals/attention.md)
**Related:** [history-free-drafter.md](history-free-drafter.md)

---

## What it is

Hybrid linear-attention models (interleaving softmax attention with linear-attention / Mamba-style blocks) compress history into **recurrent states** instead of a full KV cache. A standard speculative drafter assumes the target carries full KVs and bolts on its own smaller KV cache — which is wasted work on a hybrid target, and the drafter's view of history diverges from the target's state-space view.

ReaDy makes the drafter aware of the hybrid target: at each draft position, retrieve from the target's recurrent state the context relevant to this draft step, and condition the drafter on that retrieval instead of rebuilding history.

## How it works

**Learned query per draft position.** At each draft step the drafter emits a query vector. This query indexes into the target's **committed recurrent state** (not the full history), making retrieval content-adaptive rather than fixed-window.

**State reader.** A thin head converts the retrieved state slice into draft-space features. Think of it as "the inverse of the target's state-update rule, trained end-to-end as a projection the drafter can consume."

**Standard draft-verify loop.** The drafter proposes $k$ tokens; the target verifies them in one forward pass. Accepted tokens commit, and the drafter's query generator uses the newly-updated recurrent state on the next step.

**No parallel KV cache.** The drafter has no history of its own — all history-dependence flows through the target's recurrent state.

## Why it matters

- **Right abstraction for hybrid LA.** Prior drafters assume softmax attention; they can't exploit the target's compressed state. ReaDy makes the drafter's inductive bias match the target's.
- **Carries speculative decoding to the next-gen architecture family.** As hybrid-LA models become more common (Mamba-2, Gated DeltaNet, Jamba-style stacks), existing drafter work doesn't transfer. ReaDy is a proof-of-concept that the speculative-decoding playbook survives.
- **Same "drop draft history, read target state" idea** as DLite ([history-free-drafter.md](history-free-drafter.md)), specialized to the state-space target.

## Gotchas & tricks

- **Learned query must track target updates.** If the target's recurrent-state update rule changes (fine-tune, LoRA on linear-attention blocks), the drafter's query generator usually needs a short refresh pass.
- **Not for pure-softmax targets.** On a vanilla transformer, the recurrent state doesn't exist as a compact object; ReaDy's advantage disappears.
- **Benchmarks included dialogue, code, and math.** Gains are consistent across all three, but strongest where the recurrent state carries a long, semantically dense history (long dialogue > short code completion).

## Sources

- Paper: *ReaDy: Recurrent States as Dynamic Context for Speculative Drafting in Hybrid Linear Attention Large Language Models* — Sun et al., 2026 — [arXiv:2610.20503](https://arxiv.org/abs/2610.20503)
