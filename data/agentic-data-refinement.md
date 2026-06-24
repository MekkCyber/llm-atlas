# Agentic Data Refinement
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of passively annotating data once, train an **agent that actively tailors raw multimodal streams to the downstream training task**. The agent's policy is shaped with GRPO using rewards derived from how well the refined dataset trains the target model. Introduced as DataClaw₀ (2026), a 9B-parameter agent trained with SFT + GRPO and validated across video generation, VQA, and GUI navigation downstreams.

**Prereqs:** [_data-curation.md](_data-curation.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [quality-filtering.md](quality-filtering.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Classical data curation runs one fixed pipeline: dedup → quality filter → mixture. The pipeline is *task-blind*: it doesn't know whether the downstream model is a video generator, a VLM, or a code agent.

Agentic data refinement reframes curation as a **policy problem**. An agent reads chunks of the raw stream, chooses refinement actions (rewrite, restructure, extract procedural sub-steps, drop, link), and emits a refined sample. The policy is updated so that refinements that improve a downstream task model are reinforced.

## How it works

The training loop:

1. **SFT warm-start.** The base agent is supervised on a small set of human-refined examples covering the action vocabulary (rewrite, restructure, etc.).
2. **GRPO rollout.** For a raw chunk, the agent samples `G` candidate refinements.
3. **Downstream reward.** A small downstream model is trained / fine-tuned on each candidate refinement and evaluated on a task-specific validation set. The validation score is the reward.
4. **Group-relative advantage.** Standard GRPO: subtract the group mean, divide by std, broadcast to all tokens in the refinement.
5. **PPO-clipped policy update** with KL to the SFT reference.

DataClaw₀ adds a refinement-quality validation set (`DataClaw₀-val`) that decouples refinement quality from any single downstream task.

## Why it matters

- Multimodal post-training is **data-bottlenecked**, not compute-bottlenecked. A 9B refinement agent that surfaces procedural structure from raw video is worth more than another order of magnitude of unrefined data.
- The reward signal is **downstream task performance**, which closes the loop that one-shot curation pipelines leave open. Curation choices that look good by surface heuristics (length, fluency) but hurt downstream learning get pruned automatically.
- It is a *transferable pattern*: the same loop can target code-agent corpora, robot-trajectory corpora, GUI-navigation corpora.

## Gotchas & tricks

- The downstream-training reward is expensive — full SFT per rollout is infeasible. Practical implementations use small surrogate models (proxy LMs / small video models) for the reward and verify with full-scale training only periodically.
- **Reward hacking** is the obvious failure: the agent learns refinements that exploit the surrogate model's quirks. Mitigation is rotating surrogate seeds and including a diverse validation set.
- The action vocabulary has to be large enough to express useful refinements but small enough for GRPO to explore. DataClaw₀ uses a few-dozen-action vocabulary.

## Sources

- Paper: *DataClaw₀: Agentic Tailoring Multimodal Data from Raw Streams* — Luo, Ma, Gong, XJTU / UCAS / Tsinghua, 2026 — [arXiv:2606.21337](https://arxiv.org/abs/2606.21337).
