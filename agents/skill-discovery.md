# Unsupervised Skill Discovery for Agents
*Depth — mining reusable procedural skills from an agent's own exploration, no labels, no weight updates.*

**TL;DR:** Agentic systems improve faster from **skill libraries** (reusable procedural prompts) than from re-fine-tuning. The open problem is *where the skills come from*. DataCOPE (Song et al., 2026) runs the agent in exploration mode on unlabeled tasks, uses a self-consistency-style **verifier** to rank candidate trajectories, distills recurring high-scoring sub-procedures into the library, and at inference retrieves the matching skill into the prompt. The agent improves with zero parameter updates and zero human labels.

**Prereqs:** [_post-training](../post-training/_post-training.md)
**Related:** [rejection-sampling](../post-training/rejection-sampling.md), [agent-memory-policy](agent-memory-policy.md)

---

## What it is

A discovery loop for inference-time skill augmentation. The output is a library of *(trigger condition, procedural prompt)* entries that the agent retrieves and injects into context at inference time. Crucially, the discovery is unsupervised: no per-task labels, no human authorship.

## How it works

```
loop:
    sample task t from unlabeled task pool
    explore: generate K candidate trajectories from current agent
    verify:  score each trajectory with a verifier V (often self-consistency
             across K, or rule-based when applicable)
    cluster: high-scoring trajectories that share substructure become candidate skills
    distill: turn the cluster's shared sub-procedure into a reusable prompt template
    add to library
```

Three design choices that matter:

- **Verifier choice.** Self-consistency on terminal answers is the universal fallback when no rule verifier exists. For data analysis (DataCOPE's domain) this works because the same query can be reached via multiple analytic paths.
- **What counts as a "skill"?** A sub-trajectory that recurs across trajectories with high verifier score. Not single actions; not whole trajectories. Granularity is set by clustering bandwidth.
- **Retrieval at inference.** A small embedding-based retriever picks the skill whose trigger matches the current sub-task. Injected as procedural context, not as a tool call.

## Why it matters

- **No fine-tuning needed.** Adding skills is a library-write, not a weight-update — much faster iteration than RL post-training.
- **Composable across formats.** DataCOPE shows discovered skills generalize across task formats (different question types, different data shapes), because the procedural representation is abstract.
- **Label-free.** The whole loop runs on unlabeled task streams. The verifier replaces the label.

## Gotchas & tricks

- **Verifier reliability bounds the library quality.** If the verifier is biased, the library inherits the bias. Audit periodically against a held-out labeled set.
- **Library bloat.** Without pruning, the library accumulates near-duplicate skills. Periodic dedup by skill similarity + downstream-usage stats is essential.
- **Cold start.** Until the library is non-trivial, the agent has nothing to retrieve. Bootstrap with a small seed library or run discovery for some warmup iterations before serving.
- **Skill drift.** Skills mined against an older model checkpoint may underperform after the model is updated. Re-mine periodically.
- **Distinction from agent memory.** Skills are *general procedural* knowledge; memory is *per-episode* state. The library is shared across sessions; memory is not.

## Sources

- Paper: *Unsupervised Skill Discovery for Agentic Data Analysis* — Song et al., Zhejiang U., 2026 — [arXiv:2606.06416](https://arxiv.org/abs/2606.06416) — introduces DataCOPE for the data-analytic agent setting.
