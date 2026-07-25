# Recursive Self-Improvement (RSI) for Agents
*Depth — the verify-then-refine outer loop that turns an agent's own partial output into the next input.*

**TL;DR:** RSI in the agent sense is not a foom-scenario capability jump — it's a concrete architectural pattern: an outer loop that (i) verifies the agent's current answer, (ii) identifies what still fails, (iii) launches targeted follow-up work, and (iv) compresses the growing interaction history into a compact *improvement state* that persists across iterations. The key move is treating the agent's own partial output as *evidence*, not as prompt bloat, so long horizons stay tractable.

**Prereqs:** [../agents/README](README.md), [deep-research](deep-research.md)
**Related:** [../post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md), [../post-training/rejection-sampling](../post-training/rejection-sampling.md)

---

## What it is

An outer control loop over a working agent. Each iteration:

```
state ← compress(state, previous_iteration)
answer ← inner_agent(question, state)
audit  ← constraint_wise_verify(answer)
if audit.all_pass: return answer
state ← update(state, audit.unresolved, decisive_evidence)
```

The agent doesn't produce a new answer from scratch each iteration — it *edits* the current answer in the light of what verification exposed. The "recursive" in RSI refers to the outer loop taking its own previous output as one of its inputs.

## How it works

Three ingredients:

1. **Constraint-wise verification.** The inner-loop answer is decomposed into checkable claims, each scored independently. Verification is asymmetrically cheap: even when the answer is complex, checks are usually one-shot.
2. **Improvement state.** A structured object separate from raw history, holding *verified evidence*, *unresolved constraints*, and *ruled-out directions*. Prompted back to the inner loop as compact context rather than as full transcripts.
3. **Learned context-update tool.** Rather than a hand-written summarizer, the agent trains a tool whose job is to fold each iteration's outcome into the improvement state. Because the tool is trained end-to-end with the agent, it learns to keep exactly the fragments the next iteration will use.

Training uses long-horizon RL over the outer loop, with denser credit on *decisive steps* (where an audit flipped a wrong direction or a search resolved a hard constraint) to counter the sparsity of the final task reward.

## Why it matters

- **Turns extra compute into extra correctness.** Naive rollout extension hits diminishing returns as context grows and salience drops. RSI extracts signal from each iteration and discards the rest.
- **Cleanly separates thinking from search.** The inner agent can be any tool-using LLM; the outer loop is a discipline layered on top. Both can be improved independently.
- **Puts memory design under the RL objective.** Context compression is often ad-hoc plumbing. RSI makes it a first-class trainable component.

## Gotchas & tricks

- **The verifier is the ceiling.** If audits accept plausible-but-wrong answers, more iterations just polish the wrong one. Bias the verifier toward *refuted* by default.
- **Compression can drop the evidence you needed.** Watch for iterations that re-search facts you already had — that's a compressor failure mode, not an agent failure.
- **Don't confuse RSI with self-training.** RSI edits an *answer* within a single session; self-training edits *weights* across sessions. They compose but they aren't the same loop.
- **Reward decisive steps, not iteration count.** Rewarding "one more iteration" produces verbose non-improvers.

## Sources

- Paper: *AREX: Towards a Recursively Self-Improving Agent for Deep Research* — Lu et al., 2026 — [arXiv:2607.21461](https://arxiv.org/abs/2607.21461).
