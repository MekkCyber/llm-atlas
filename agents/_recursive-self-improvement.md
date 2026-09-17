# Recursive Self-Improvement (RSI)

*Taxonomy — training loops in which the agent's outputs improve itself, its harness, or both.*

**TL;DR:** RSI names any loop where an AI system uses its own experience and feedback to make persistent changes that improve future capability. In 2026 the term splits into three practical species — *harness-only* evolution, *model-only* RL post-training, and *coupled* alternation of the two — with an emerging fourth, *meta-improvement*, where the system rewrites the improvement process itself. The modern default for research systems is coupled: freeze the model, evolve the harness, then RL the model under the improved harness, and repeat.

**Related taxonomies:** [_rl](../post-training/_rl.md)
**Depth files covered here:** [harness-evolution](harness-evolution.md) · [recursive-in-recursive-si](recursive-in-recursive-si.md) · [modular-harness](modular-harness.md)

---

## The problem

Post-training gains have to come from somewhere. Human-labeled data is expensive and slow, and RL-from-verifier alone plateaus once the model saturates the environment it trains in. RSI recycles the model's *own* execution — successful and failed trajectories, rubric feedback, tool logs — into persistent changes, letting the system chase its own frontier without a new human-annotated dataset for each round.

## The shared pattern

All RSI variants share the same three-part loop:

1. **Execute.** The agent runs on tasks with its current model + harness.
2. **Attribute.** Successes and failures are traced to something modifiable — a prompt, a tool, a routing rule, or the model's weights.
3. **Update.** That something is edited (harness edit) or trained (RL / SFT on curated trajectories).

Variants differ in *what is modifiable* and *how attribution flows* between the harness and the model.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [harness-evolution](harness-evolution.md) | Model frozen; edit prompts, tools, routing rules from trajectories | Cheap, no gradient step, but capped by the frozen model | Rapid iteration on scaffold; deploy-time-only changes |
| [recursive-in-recursive-si](recursive-in-recursive-si.md) | Alternate inner harness evolution with outer model RL | Compounding gains but coordination cost is high | Coupled model + scaffold improvement |
| [modular-harness](modular-harness.md) | Split the harness into 5 modules and evolve each independently | Attribution is cleaner; integration must reconcile them | Generalization across benchmarks and models |
| model-only RL (no depth file) | Freeze the harness, RL the model on rollouts | Simple, but scaffold stays as-is | Existing harness already strong |
| meta-improvement (no depth file yet) | System edits the improvement rule itself | Rare; theoretical for now | Long horizons where the update rule matters |

## How to choose

- **Prototype fast → harness-evolution.** No gradient step, no GPU cluster; iterate on prompts and tools in hours.
- **Long-horizon coding / SWE tasks → modular-harness.** Attribution across a monolithic harness is the bottleneck; the five-module split from ModularRSI is the current SOTA on cross-benchmark transfer.
- **Coupled research system → recursive-in-recursive-si.** When you own both the harness and the training loop, alternating inner (harness) and outer (model RL) rounds compounds gains that neither alone would achieve.
- **Deployed frontier model with fixed weights →** harness evolution only; model RL is off the table.

## Adjacent but distinct

- [long-cot-rl](../post-training/reasoning/long-cot-rl.md) — RL over long reasoning traces; a component of the model-RL step, not RSI by itself.
- [rl-prompt-curation](../post-training/rl-prompt-curation.md) — data-side of RL, generalized to harness evolution here.
- [rejection-sampling](../post-training/rejection-sampling.md) — trajectory filtering, a common substrate for both harness- and model-side updates.

## Sources

- *The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement* — Duan et al., 2026 — [arXiv:2609.11873](https://arxiv.org/abs/2609.11873).
- *ScienceBuddy: Recursive-in-Recursive Self-Improvement for Interactive Scientific Agents* — Xue et al., 2026 — [arXiv:2609.17523](https://arxiv.org/abs/2609.17523).
- *ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement* — Wu et al., 2026 — [arXiv:2609.14857](https://arxiv.org/abs/2609.14857).
