# Memory self-supervision (MemTrain)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** MemTrain trains the context-memory capability of long-horizon LLM agents *self-supervisedly* over unlabeled Wikipedia, sidestepping the cost and narrowness of annotated memory benchmarks. Two coupled proxy objectives — masked-entity reconstruction after multi-step memory updates, and intermediate memory-recall — are co-optimized with GRPO. Improves long-text QA by up to +17.67 points over end-to-end RL baselines. From Li et al., 2026.

**Prereqs:** [README.md](README.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md), [streaming-multi-agent.md](streaming-multi-agent.md)

---

## What it is

A memory-agent is an LLM equipped with a per-session memory module: a fixed-size buffer (text or latent) that is updated across interaction steps so that information from earlier steps remains accessible later. Training such agents has been bottlenecked by data — high-quality long-horizon memory annotations are expensive, and the resulting training sets cover narrow scenarios.

MemTrain reframes the problem: any unlabeled long document is implicit supervision for memory, because the model must *carry* information across chunks to predict things later in the document. This is the long-horizon analogue of MLM for plain language pretraining.

## How it works

For an unlabeled Wikipedia article split into ordered chunks $c_1, c_2, \ldots, c_n$:

1. **Masked-entity targets.** Identify entities in the article. Construct targets that probe whether the agent's memory retains those entities across the full sequence.
2. **End-to-end masked reconstruction.** The agent reads each chunk, updates memory, and at the end must reconstruct the masked entities. This is the *terminal* signal — directly tests whether memory carried information forward.
3. **Intermediate memory-recall objective.** Periodically interrupt the rollout and require the agent to reconstruct historical chunk content from the *current* memory state. This prevents shortcut behaviors where the model defers all useful work to the final step.
4. **GRPO optimization.** Both signals are scored by a verifier; advantages are computed group-relative; the policy and memory-update mechanism are trained jointly.

The two objectives are *coupled*: the intermediate-recall term anchors the memory state, while the terminal masked-reconstruction term ensures the memory remains useful at the end.

## Why it matters

- **Removes the data bottleneck.** Memory training can now scale on plain text, the same way pretraining does. Wikipedia is just the start — any long-form corpus works.
- **Better than annotated benchmarks.** Reported +17.67 pts on long-text QA and +10.58 pts on search-based QA vs end-to-end RL on annotated memory tasks. Self-supervision wins on diversity.
- **Compatible with any memory architecture.** The objectives are agnostic to whether memory is a text scratchpad, learnable token bank, or compressed latent state.

## Gotchas & tricks

- **Entity selection drives signal.** Trivial entities (stop words, dates the agent could re-derive) waste the objective; rare, specific, mid-document entities are the most informative.
- **Intermediate-recall placement.** Probing too early gives nothing to recall; too late and the shortcut returns. Spacing probes through the rollout matters.
- **GRPO scale.** Joint optimization over policy + memory update with group rollouts is compute-heavy; consider batched memory snapshots to share forward passes.
- **Doesn't replace SFT seeding.** Pure self-supervised memory training from scratch is harder than starting from a model that already has basic instruction-following.

## Sources

- Paper: *MemTrain: Self-Supervised Context Memory Training* — Li, Xing, Wang, Deng, Tang, 2026 — [arXiv:2606.03197](https://arxiv.org/abs/2606.03197).
- Related: GRPO (Shao et al., 2024); MLM as a self-supervised proxy for representation learning.
